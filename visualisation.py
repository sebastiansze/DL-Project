import os
import pickle
# from pathos import multiprocessing as mp
import itertools
from tqdm import tqdm
import numpy as np

import colorsys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
# from IPython.display import HTML
import imageio

from GameLogic import Game, Point


def print_layers(layers, fill='\u2590\u2588\u258C'):
    if layers.ndim == 2:
        layers = np.expand_dims(layers, 0)

    if layers.ndim != 3:
        raise ValueError('Invalid number of dimensions')

    print('\u250f' + '\u2501' * (layers.shape[2] * 3) + '\u2513')
    for i, layer in enumerate(layers):
        if i > 0:
            print('\u2523' + '\u2501' * (layers.shape[2] * 3) + '\u252B')
        for row in layer:
            print('\u2503', end='')
            for cell in row:
                if cell:
                    print(fill, end='')
                else:
                    print('   ', end='')
            print('\u2503')
    print('\u2517' + '\u2501' * (layers.shape[2] * 3) + '\u251B')


def fig_to_data(fig):
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()

    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)

    buf = np.roll(buf, 3, axis=2)
    return buf


class Visualisation:
    def __init__(self, input_maps, map_size, agent_count, view_padding, view_reduced=False, truth_obstacles=None):
        self._map_size_x = map_size[0]
        self._map_size_y = map_size[1]
        self._view_padding = view_padding
        self._view_size_x = view_padding[0] + 1 + view_padding[1] if view_reduced else 0
        self._view_size_y = view_padding[2] + 1 + view_padding[3] if view_reduced else 0
        self._view_reduced = view_reduced
        self._agent_count = agent_count
        self.time_steps = len(input_maps)
        self._next_step = False

        input_maps = np.array(input_maps)
        if not view_reduced:
            self._input_maps = np.reshape(input_maps,
                                          (self.time_steps, agent_count, self._map_size_x, self._map_size_y))
            self._full_maps = self._input_maps
        else:
            # Separate the real maps from further position information (current and aim position -> c_x, c_y, a_x, a_y)
            self._input_maps = np.reshape(input_maps[:, :, 0:self._view_size_x * self._view_size_y],
                                          (self.time_steps, agent_count, self._view_size_x, self._view_size_y))
            self._current_pos = (input_maps[:, :, -4:-2] * map_size).round().astype('int64')
            self._aim_pos = (input_maps[:, :, -2:] * map_size).round().astype('int64')
            if np.unique(self._current_pos, axis=0).shape[0] > 1:
                print('Warning: Aim positions changed over time')

            # Rebuild the full maps of the environment and start with a empty matrix
            # The size of a single map is padded to apply agents field of view also if a agent stands close to a corner
            # Shape: (time_steps, agent_counts, X, Y)
            self._full_maps = np.zeros((self.time_steps,
                                        self._agent_count,
                                        view_padding[0] + self._map_size_x + view_padding[1],
                                        view_padding[2] + self._map_size_y + view_padding[3]))

            # Create a list on indices for full_maps to put input_maps into it later
            # Shape: (4, time_steps * agent_counts * map_size_x * map_size_y)
            indices = np.array(list(itertools.product(np.arange(self.time_steps),
                                                      np.arange(agent_count),
                                                      np.arange(self._view_size_x),
                                                      np.arange(self._view_size_y)))).T

            # Add the current positions as offset to the indices to bring the smaller input_maps to the right positions
            indices[2] += np.repeat(self._current_pos[:, :, 0], self._view_size_x * self._view_size_y)  # x offset
            indices[3] += np.repeat(self._current_pos[:, :, 1], self._view_size_x * self._view_size_y)  # y offset

            # Fill the full maps with values from the input maps at the right positions
            self._full_maps[indices[0], indices[1], indices[2], indices[3]] = self._input_maps.flatten()

            # Crop out the padding of the full maps
            self._full_maps = self._full_maps[:, :, view_padding[2]:-view_padding[3], view_padding[0]:-view_padding[1]]

        # Obstacles
        self._obstacle_maps = (self._full_maps == 0.25)
        if not np.all(np.isin(np.count_nonzero(self._obstacle_maps, axis=(0, 1)), [0, self.time_steps * agent_count])):
            print('Warning: Positions of obstacles changed over time or are different for different agents')
        if isinstance(truth_obstacles, type(None)):
            self._obstacle_pos = np.argwhere(np.any(self._obstacle_maps, axis=(0, 1)))
        else:
            self._obstacle_pos = truth_obstacles

        # Others Position
        self._others_maps = (self._full_maps == 0.5)

        # Aim Positions
        self._aim_maps = (self._full_maps == 0.75)
        if not np.all(np.isin(np.count_nonzero(self._aim_maps, axis=0), [0, self.time_steps])):
            print('Warning: Aim maps changed over time')

        # Current Positions
        self._current_maps = (self._full_maps == 1.0)
        if np.any(np.count_nonzero(self._current_maps, axis=(2, 3)) > 1):
            print('Warning: At least one time step there are several positions for one or more agents')
        elif np.any(np.count_nonzero(self._current_maps, axis=(2, 3)) < 1):
            print('Warning: At least at one time step for one or more agents the positions are missing')

        # Agent status includes 'aim achieved' (a), 'self inflicted accident' (s), 'third-party fault accident' (3)
        # and 'time out' (t)
        self._agents_conditions = np.zeros(self._agent_count, dtype=np.dtype('U1'))

        # Color
        self._color_hue_offset = np.random.uniform()

        # Latex Settings
        custom_preamble = {
             "text.usetex": True,
            "text.latex.preamble": [
                r"\usepackage{amsmath}",  # for the align enivironment
            ],
        }
        plt.rcParams.update(custom_preamble)
        mpl.use('TkAgg')

    def get_map_for_agent(self, time_step=-1, agent=0, plot_input=False):
        """
        Get map for agent X's point of view including: obstacles, own aim position,
        own current position (, own next position), others current position (, others next position).
        :param agent: number of agent
        :param view_filed: size of view field
        :return: map as boolean array with shape [4 or 6, size_x, size_y]
        """
        obstacles = self._obstacle_maps[time_step, agent]
        aim_map = self._aim_maps[time_step, agent]
        cur_map = self._current_maps[time_step, agent]
        # nxt_map = ... TODO: Next step
        others_cp = self._others_maps[time_step, agent]
        full_map = self._full_maps[time_step, agent]
        input_map = self._input_maps[time_step, agent]

        if plot_input:
            agent_map = [obstacles, aim_map, cur_map, others_cp, full_map, input_map]  # TODO: Next step
        else:
            agent_map = [obstacles, aim_map, cur_map, others_cp]  # TODO: Next step

        # TODO: Next step
        # if self._next_step:
        #     others_np = np.any(self.get_filtered_map(layer='n'), axis=0)  # next positions of other agents
        #     others_np = others_np & ~self.get_filtered_map(agent=agent, layer='n')  # subtract own next position
        #     agent_map = np.concatenate((agent_map, others_np))

        return agent_map

    def _get_plot_color(self, agent_index, next_step=False):
        hue = agent_index / self._agent_count + self._color_hue_offset
        saturation = 1.0 if not next_step else 0.1
        value = 0.7 if not next_step else 0.9
        return colorsys.hsv_to_rgb(hue, saturation, value)

    def _plot_map_border(self, ax):
        border = patches.Rectangle((0, 0), self._map_size_y, self._map_size_x, linewidth=5, edgecolor='black',
                                   facecolor='none')
        ax.add_patch(border)

    def _plot_view_border(self, ax, pos):
        start = (pos[1] - self._view_padding[1],
                 self._map_size_x - pos[0] - 1 - self._view_padding[3])
        border = patches.Rectangle(start, self._view_size_y, self._view_size_x, linewidth=2, edgecolor='grey',
                                   facecolor='none')
        ax.add_patch(border)

    def _plot_rect_at_pos(self, ax, x, y, color):
        rect = patches.Rectangle((y, self._map_size_x - x - 1), 1, 1, linewidth=0,
                                 edgecolor='none', facecolor=color)
        ax.add_patch(rect)

    def _plot_label(self, ax, x, y, text, color):
        x = self._map_size_x - x - 1
        prop = FontProperties(family='monospace', weight='black')
        tp = TextPath((y, x), text, prop=prop, size=1)
        polygon = tp.to_polygons()
        for a in polygon:
            patch = patches.Polygon(a, facecolor=color, edgecolor='black', linewidth=1, zorder=10)
            ax.add_patch(patch)

    def _plot_layer(self, ax, layer, color, plot_view_filed=False, curr_pos=None):
        # Plot Layer
        for x in range(self._map_size_x):
            for y in range(self._map_size_y):
                if layer[x, y]:
                    self._plot_rect_at_pos(ax, x, y, color)

        # Plot view field
        if plot_view_filed:
            self._plot_view_border(ax, curr_pos)

        # Plot Border
        self._plot_map_border(ax)

        ax.set_ylim(0, self._map_size_x)
        ax.set_xlim(0, self._map_size_y)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    def _plot_heatmap(self, ax, map):
        ax.imshow(map, cmap='hot', interpolation='nearest', vmin=0, vmax=1)

        # ax.set_ylim(0, self._size_x)
        # ax.set_xlim(0, self._size_y)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    def _plot_info(self, ax, time_step, i_game=None):
        text = r'\begin{align*}'
        if not isinstance(i_game, type(None)):
            text += r'i_{{game}}&={}\\'.format(i_game)
        text += r't&={}\\'.format(time_step)
        text += r'size_{{map}}&=\left[{}\times{}\right]\\'.format(self._map_size_x, self._map_size_y)
        if self._view_reduced:
            text += r'size_{{view}}&=\left[{}\times{}\right]\\'.format(self._view_size_x, self._view_size_y)
        text += r'\end{align*}'

        ax.text(0.3, 0.65, text, fontsize=17, ha='left', va='center')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    def _plot_overview(self, ax, time_step=-1, plot_agent_status=True, plot_path=True):
        # Obstacles
        for x, y in self._obstacle_pos:
            self._plot_rect_at_pos(ax, x, y, 'black')

        # Agents fields
        for i_agent in range(self._agent_count):
            if self._view_reduced:
                start_pos = [self._current_pos[0, i_agent]]
                cur_pos = [self._current_pos[time_step, i_agent]]
                aim_pos = [self._aim_pos[time_step, i_agent]]
            else:
                start_pos = np.argwhere(self._current_maps[0, i_agent])
                cur_pos = np.argwhere(self._current_maps[time_step, i_agent])
                aim_pos = np.argwhere(self._aim_maps[time_step, i_agent])

            # nxt_map = ... TODO: Next step

            color = self._get_plot_color(i_agent, next_step=False)
            color_next = self._get_plot_color(i_agent, next_step=True)

            # Plot next position
            # TODO: Next step
            # if self._next_step:
            #     self._plot_layer(ax, nxt_map, color_next)

            # Plot current position
            for x, y in cur_pos:
                self._plot_rect_at_pos(ax, x, y, color)

            # Plot path
            if plot_path:
                hist = np.where(self._current_maps[0:time_step + 1, i_agent])
                offset = (1 / (self._agent_count + 1) * (i_agent + 1) * 0.5) - 0.25
                x = hist[2] + 0.5 + offset
                y = self._map_size_x - hist[1] - 0.5 + offset
                ax.plot(x, y, '-', color=color, zorder=0)

            # Plot start position
            # if start_pos is not None and start_pos[0].shape[0] != 0:
            for x, y in start_pos:
                self._plot_label(ax, x-0.15, y+0.2, "S", color)
            # else:
            #     print('Warning: No start position')

            # Plot aim position
            for x, y in aim_pos:
                self._plot_label(ax, x-0.15, y+0.2, "E", color)

            # Plot agent status
            if plot_agent_status:
                for status, symbol in zip(['a', 's', '3', 't'], ['\u2713', '\u2717', '\u2717', '\u2717']):  # \u2620
                    if self._agents_conditions[i_agent] == status:
                        for x, y in self._current_maps[time_step, i_agent]:
                            self._plot_label(ax, x-0.15, y+0.2, symbol, 'black')

        # Plot Border
        self._plot_map_border(ax)

        ax.set_ylim(0, self._map_size_x)
        ax.set_xlim(0, self._map_size_y)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    def _plot_all(self, fig, time_step=-1, plot_agent_status=True, plot_path=True, plot_input=False, i_game=None):
        # Create outer grid
        outer = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1, width_ratios=[0.382, 0.618])
        outer.update(left=0.01, right=0.99, top=0.95, bottom=0.01)

        # Plot overview and info on the left side
        left_grid = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0],
                                                     wspace=0.1, hspace=0.1, width_ratios=[1], height_ratios=[1, 4, 1])
        ax = plt.subplot(left_grid[1])
        self._plot_overview(ax, time_step=time_step, plot_agent_status=plot_agent_status, plot_path=plot_path)
        ax.set_title('Overview', fontsize=15)
        fig.add_subplot(ax)
        ax = plt.subplot(left_grid[2])
        self._plot_info(ax, time_step, i_game)
        fig.add_subplot(ax)

        # Plot Layers
        if self._next_step:
            nr_layers = 6
            layer_names = ['Obstacles', 'Aim', 'Agent\'s\nCurrent Pos.', 'Agent\'s\nNext Pos.',
                           'Others\nCurrent Pos.', 'Others\nNext Pos.']
        else:
            nr_layers = 4
            layer_names = ['Obstacles', 'Aim', 'Agent\'s\nPosition', 'Others\nPosition']
        if plot_input:
            nr_layers += 2
            layer_names.append('Full Map\nNet Input')
            layer_names.append('Reduced\nNet Input')
        agents_grid = gridspec.GridSpecFromSubplotSpec(self._agent_count, nr_layers, subplot_spec=outer[1],
                                                       wspace=0.1, hspace=0.1)
        for i_agent in range(self._agent_count):
            layers = self.get_map_for_agent(time_step=time_step, agent=i_agent, plot_input=plot_input)
            for i_layer, layer in enumerate(layers):
                i_grid = i_agent * nr_layers + i_layer
                ax = plt.subplot(agents_grid[i_grid])
                if plot_input and i_layer + 2 >= nr_layers:
                    self._plot_heatmap(ax, layer)
                else:
                    color = self._get_plot_color(i_agent)
                    if self._view_reduced:
                        self._plot_layer(ax, layer, color, plot_view_filed=True,
                                         curr_pos=self._current_pos[time_step, i_agent])
                    else:
                        self._plot_layer(ax, layer, color)

                # layer label
                if ax.is_first_row():
                    ax.set_xlabel(layer_names[i_layer], fontsize=15)
                    ax.xaxis.set_label_position('top')

                # agent label
                if ax.is_first_col():
                    ax.set_ylabel('Agent {}'.format(i_agent), fontsize=15)

                fig.add_subplot(ax)

        plt.subplots_adjust(wspace=0, hspace=0)

        return fig

    def plot_layer(self, layer, block=True, save_as=None):
        """
        Shows a single layer
        :param save_as:
        :param block:
        :param layer: number of agent
        :return:
        """
        # Disable tools and create figure and axes
        mpl.rcParams['toolbar'] = 'None'
        fig, ax = plt.subplots(1, figsize=(5, 5))

        # Plot Layer
        self._plot_layer(ax, layer, 'black')

        if save_as:
            fig.savefig(save_as)
            plt.close(fig)
        else:
            plt.show(block=block)

    def plot_overview(self, time_step=-1, plot_agent_status=True, plot_path=True, block=True, save_as=None):
        """
        Shows an overview for humans
        :return:
        """
        # Disable tools and create figure and axes
        mpl.rcParams['toolbar'] = 'None'
        fig, ax = plt.subplots(1, figsize=(5, 5))

        # Plot overview
        self._plot_overview(ax, time_step=time_step, plot_agent_status=plot_agent_status, plot_path=plot_path)

        if save_as:
            fig.savefig(save_as)
            plt.close(fig)
        else:
            plt.show(block=block)

    def plot_all(self, time_step=-1, plot_agent_status=True, plot_path=True, plot_input=False,
                 block=True, save_as=None, i_game=None):
        """
        Shows an overview and all layers for each single agent in one plot
        :return:
        """
        # Disable tools and create figure, axes and outer grid
        mpl.rcParams['toolbar'] = 'None'
        img_width = 1920
        img_height = 1080
        dpi = 120
        fig = plt.figure(figsize=(img_width/dpi, img_height/dpi), dpi=dpi)
        fig = self._plot_all(fig, time_step=time_step, plot_agent_status=plot_agent_status,
                             plot_path=plot_path, plot_input=plot_input, i_game=i_game)
        fig.set_size_inches(img_width/dpi, img_height/dpi)

        if save_as:
            fig.savefig(save_as, dpi=dpi)
            plt.close(fig)
        else:
            plt.show(block=block)

    def save_all_as_video(self, dt, i_game, plot_agent_status=True, plot_path=True, plot_input=False):
        img_width = 1920
        img_height = 1080
        dpi = 120

        plt.ioff()  # prevent matplotlib from running out of memory

        def draw_frame(ts):
            fig, ax = plt.subplots(figsize=(img_width / dpi, img_height / dpi), dpi=dpi)
            fig = self._plot_all(fig, time_step=ts, plot_agent_status=plot_agent_status,
                                 plot_path=plot_path, plot_input=plot_input, i_game=i_game)
            fig.set_size_inches(img_width / dpi, img_height / dpi)
            data = fig_to_data(fig)
            # fig.clf()
            # plt.clf()
            plt.close(fig)
            return data

        # Make the pool of workers
        # pool = mp.ProcessingPool(mp.cpu_count() - 1)

        # Start multithreading
        # frame_array = list(tqdm(pool.imap(draw_frame, np.arange(self.time_steps)), total=self.time_steps))
        frame_array = [draw_frame(ts) for ts in tqdm(range(self.time_steps))]

        # Close the pool and wait for the work to finish
        # pool.close()
        # pool.join()

        directory = os.path.join('img', dt)

        # Check if directory for images exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        w = imageio.get_writer(os.path.join(directory, f'{dt}_game_{i_game}.mp4'),
                               fps=4, quality=6, macro_block_size=20)
        for i in range(len(frame_array)):
            w.append_data(frame_array[i])
        w.close()

    def save(self, dt, i_game):
        directory = os.path.join('viz', dt)

        # Check if directory for viz exists
        if not os.path.exists(directory):
            os.makedirs(directory)
        f = open(os.path.join(directory, f'{dt}_game_{i_game}.viz'), 'wb')
        pickle.dump(self, f, 2)
        f.close()

    @staticmethod
    def load(path):
        f = open(path, 'rb')
        viz = pickle.load(f)
        f.close()
        return viz


class Helpers:
    def __init__(self, env: Game):
        self.env = env

    def show_hist_map(self, hist):
        for i in range(len(hist)):
            if all(elm < self.env.board_size[0] for elm in hist[int(i)]):
                h = hist[i].astype(np.int)

                if np.max(hist[i]) == 1:
                    plt.imshow(hist[i].reshape(self.env.board_size[0], self.env.board_size[1]), cmap='hot',
                               interpolation='nearest')
                else:
                    plt.imshow(np.zeros((self.env.board_size[0], self.env.board_size[1])))
                plt.show()

    def show_reward(self, h, elem_max=True):
        m = np.zeros(self.env.board_size)
        for i in range(self.env.board_size[0]):
            for j in range(self.env.board_size[1]):
                reward, _ = self.env.get_reward_for_position(Point(i, j))
                if reward > 30 and elem_max:
                    m[i, j] = 100
                else:
                    m[i, j] = reward

        return m.astype(np.int)

    def show_hist_ani(self, hist):
        frame_array = []
        for i in range(len(hist)):
            if all(elm < self.env.board_size[0] for elm in hist[int(i)]):
                h = hist[i].astype(np.int)
                fig = plt.figure()
                if np.max(hist[i]) == 1:
                    plt.imshow(hist[i].reshape(self.env.board_size[0], self.env.board_size[1]), cmap='hot',
                               interpolation='nearest')
                else:
                    plt.imshow(np.zeros((self.env.board_size[0], self.env.board_size[1])))
                frame_array.append(fig_to_data(fig))
                plt.close()
        #

        w = imageio.get_writer('output.mp4', fps=6, quality=6)
        for i in range(len(frame_array)):
            w.append_data(frame_array[i])
        w.close()

    # def playVideo(self, path):
    #     before = """<video width="864" height="576" controls><source src="""
    #     end = """ type="video/mp4"></video>"""
    #     return HTML(before + path + end)


if __name__ == "__main__":
    viz_file_name = "2020-07-19-21-46-31"
    viz = Visualisation.load(f'{viz_file_name}.viz')
    path = os.path.join('img', f'{viz_file_name}.mp4')
    print(f'Generate video {path}...')
    viz.save_all_as_video(plot_input=True, save_as=path)
