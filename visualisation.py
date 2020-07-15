import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
# from IPython.display import HTML
import imageio
import colorsys
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


class Visualisation:
    def __init__(self, game, size_x, size_y):
        game = np.reshape(game, (len(game), size_y, size_x))
        print_layers(game == 0.3)
        print(game.shape)

        self._agent_count = 1
        self._size_x = size_x
        self._size_y = size_y
        self._next_step = False

        # History
        self._hist = np.zeros((0, self._agent_count, 2))  # shape: (time steps, agent_count, 2) -> 2 for x & y
        self._hist_next = np.zeros((0, self._agent_count, 2))  # shape: (time steps, agent_count, 2) -> 2 for x & y

        # Agent status includes 'aim achieved' (a), 'self inflicted accident' (s), 'third-party fault accident' (3)
        # and 'time out' (t)
        self._agents_conditions = np.zeros(self._agent_count, dtype=np.dtype('U1'))

        # Color
        self._color_hue_offset = np.random.uniform()

    def get_filtered_map(self, agent=None, layer=None):
        """
        Returns a filtered map with only wanted layers
        :param agent: number of agent, None for all agents
        :param layer: 'o' for obstacles, 'a' for aim positions, 'c' for current positions,
        'n' for next positions, None for all layers
        :return: filtered map as boolean array with shape (requested layers, size_x, size_y)
        """
        return np.zeros((self._size_x, self._size_y))
        # return self._map[self._layer_filter(agent, layer)]

    def get_map_for_agent(self, agent, view_filed=None):
        """
        Get map for agent X's point of view including: obstacles, own aim position,
        own current position (, own next position), others current position (, others next position).
        :param agent: number of agent
        :param view_filed: size of view field
        :return: map as boolean array with shape [4 or 6, size_x, size_y]
        """
        obstacles = self.get_filtered_map(layer='o')
        a_c_n_pos = self.get_filtered_map(agent=agent)
        others_cp = np.any(self.get_filtered_map(layer='c'), axis=0)  # current positions of other agents
        others_cp = others_cp & ~self.get_filtered_map(agent=agent, layer='c')  # subtract own current position

        # TODO: field of view
        # others_cp = others_cp & ...
        agent_map = np.concatenate((obstacles, a_c_n_pos, others_cp))

        if self._next_step:
            others_np = np.any(self.get_filtered_map(layer='n'), axis=0)  # next positions of other agents
            others_np = others_np & ~self.get_filtered_map(agent=agent, layer='n')  # subtract own next position
            # TODO: field of view
            # others_np = others_np & ...
            agent_map = np.concatenate((agent_map, others_np))

        return agent_map

    def get_positions(self, agent=None, layer=None):
        return [0, 0]  # TODO

    def get_start_positions(self, agent=None):
        """
        Returns array with coordinates of first position a given agent or all agents
        :param agent: number of agent, None for all agents
        :return: array with shape [2, 0] if agent is defined or [2, agent_count] for all agents
        """
        if self._hist.shape[0] == 0:
            return None
        if agent is None:
            return self._hist[0]
        else:
            return self._hist[0, agent]

    @staticmethod
    def _plot_label(ax, x, y, text, color):
        prop = FontProperties(family='monospace', weight='black')
        tp = TextPath((x, y), text, prop=prop, size=1)
        polygon = tp.to_polygons()
        for a in polygon:
            patch = patches.Polygon(a, facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(patch)

    @staticmethod
    def _plot_border(ax, size_x, size_y):
        border = patches.Rectangle((0, 0), size_y, size_x, linewidth=5, edgecolor='black',
                                   facecolor='none')
        ax.add_patch(border)

    def _get_plot_color(self, agent_index, next_step=False):
        hue = agent_index / self._agent_count + self._color_hue_offset
        saturation = 1.0 if not next_step else 0.1
        value = 0.7 if not next_step else 0.9
        return colorsys.hsv_to_rgb(hue, saturation, value)

    def _plot_layer(self, ax, layer, color):
        # Plot Layer
        for x in range(self._size_x):
            for y in range(self._size_y):
                if layer[x, y]:
                    rect = patches.Rectangle((y, self._size_x - x - 1), 1, 1, linewidth=0,
                                             edgecolor='none', facecolor=color)
                    ax.add_patch(rect)

        # Plot Border
        self._plot_border(ax, self._size_x, self._size_y)

        ax.set_ylim(0, self._size_x)
        ax.set_xlim(0, self._size_y)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    def _plot_overview(self, ax, plot_agent_status=True, plot_path=True):
        # Obstacles
        obstacles = self.get_filtered_map(layer='o')[0]
        self._plot_layer(ax, obstacles, 'black')

        # Agents fields
        for i_agent in range(self._agent_count):
            start_pos = self.get_start_positions(agent=i_agent)
            a_c_n_map = self.get_filtered_map(agent=i_agent)

            color = self._get_plot_color(i_agent, next_step=False)
            color_next = self._get_plot_color(i_agent, next_step=True)

            # Plot next position
            if self._next_step:
                self._plot_layer(ax, a_c_n_map[2], color_next)

            # Plot current position
            self._plot_layer(ax, a_c_n_map[1], color)

            # Plot path
            if plot_path:
                offset = (1 / (self._agent_count + 1) * (i_agent + 1) * 0.5) - 0.25
                x = self._hist[:, i_agent, 1] + 0.5 + offset
                y = self._size_x - self._hist[:, i_agent, 0] - 0.5 + offset
                ax.plot(x, y, '-', color=color, zorder=0)

            # Plot start position
            if start_pos is not None:
                x = start_pos[1] + 0.2
                y = self._size_x - start_pos[0] - 1 + 0.15
                self._plot_label(ax, x, y, "S", color)

            # Plot aim position
            for y, x in self.get_positions(agent=i_agent, layer='a'):
                x = x + 0.2
                y = self._size_x - y - 1 + 0.15
                self._plot_label(ax, x, y, "E", color)

            # Plot agent status
            if plot_agent_status:
                for status, symbol in zip(['a', 's', '3', 't'], ['\u2713', '\u2717', '\u2717', '\u2717']):  # \u2620
                    if self._agents_conditions[i_agent] == status:
                        for y, x in self.get_positions(agent=i_agent, layer='c'):
                            x = x + 0.2
                            y = self._size_x - y - 1 + 0.15
                            self._plot_label(ax, x, y, symbol, 'black')

        # Plot Border
        self._plot_border(ax, self._size_x, self._size_y)

        ax.set_ylim(0, self._size_x)
        ax.set_xlim(0, self._size_y)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

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

    def plot_overview(self, plot_agent_status=True, block=True, save_as=None, plot_path=True):
        """
        Shows an overview for humans
        :return:
        """
        # Disable tools and create figure and axes
        mpl.rcParams['toolbar'] = 'None'
        fig, ax = plt.subplots(1, figsize=(5, 5))

        # Plot overview
        self._plot_overview(ax, plot_agent_status=plot_agent_status, plot_path=plot_path)

        if save_as:
            fig.savefig(save_as)
            plt.close(fig)
        else:
            plt.show(block=block)

    def plot_all(self, plot_agent_status=True, block=True, save_as=None, plot_path=True):
        """
        Shows an overview and all layers for each single agent in one plot
        :return:
        """
        # Disable tools and create figure, axes and outer grid
        mpl.rcParams['toolbar'] = 'None'
        fig = plt.figure(figsize=(17, 10))
        outer = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1, width_ratios=[0.382, 0.618])
        outer.update(left=0.01, right=0.99, top=0.95, bottom=0.01)

        # Plot overview on the left side
        ax = plt.Subplot(fig, outer[0])
        self._plot_overview(ax, plot_agent_status=plot_agent_status, plot_path=plot_path)
        ax.set_title('Overview', fontsize=15)
        fig.add_subplot(ax)

        # Plot Layers
        if self._next_step:
            nr_layers = 6
            layer_names = ['Obstacles', 'Aim', 'Agent\'s\nCurrent Pos.', 'Agent\'s\nNext Pos.',
                           'Others\nCurrent Pos.', 'Others\nNext Pos.']
        else:
            nr_layers = 4
            layer_names = ['Obstacles', 'Aim', 'Agent\'s Pos.', 'Others Pos.']
        agents_grid = gridspec.GridSpecFromSubplotSpec(self._agent_count, nr_layers, subplot_spec=outer[1],
                                                       wspace=0.1, hspace=0.1)
        for i_agent in range(self._agent_count):
            layers = self.get_map_for_agent(i_agent)
            for i_layer, layer in enumerate(layers):
                i_grid = i_agent * nr_layers + i_layer
                ax = plt.Subplot(fig, agents_grid[i_grid])
                color = self._get_plot_color(i_agent)
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

        if save_as:
            fig.savefig(save_as)
            plt.close(fig)
        else:
            plt.show(block=block)


class Helpers:
    def __init__(self,  env: Game):
        self.env = env

    def show_hist_map(self, hist):
        for i in range(len(hist)):
            if all(elm < self.env.board_size[0] for elm in hist[int(i)]):
                h = hist[i].astype(np.int)

                if np.max(hist[i]) == 1:
                    plt.imshow(hist[i].reshape(self.env.board_size[0], self.env.board_size[1]), cmap='hot', interpolation='nearest')
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

    def fig_to_data(self, fig):
        fig.canvas.draw()

        w, h = fig.canvas.get_width_height()

        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf.shape = (h, w, 3)

        buf = np.roll(buf, 3, axis=2)
        plt.close()
        return buf

    def show_hist_ani(self, hist):
        frame_array = []
        for i in range(len(hist)):
            if all(elm < self.env.board_size[0] for elm in hist[int(i)]):
                h = hist[i].astype(np.int)
                fig = plt.figure()
                if np.max(hist[i]) == 1:
                    plt.imshow(hist[i].reshape(self.env.board_size[0], self.env.board_size[1]), cmap='hot', interpolation='nearest')
                else:
                    plt.imshow(np.zeros((self.env.board_size[0], self.env.board_size[1])))
                frame_array.append(self.fig_to_data(fig))
        #

        w = imageio.get_writer('output.mp4', fps=6, quality=6)
        for i in range(len(frame_array)):
            w.append_data(frame_array[i])
        w.close()

    # def playVideo(self, path):
    #     before = """<video width="864" height="576" controls><source src="""
    #     end = """ type="video/mp4"></video>"""
    #     return HTML(before + path + end)