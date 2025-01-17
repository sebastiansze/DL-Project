import os
import pickle
import argparse
import itertools
from tqdm import tqdm
from datetime import datetime

import numpy as np

import colorsys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
import imageio

from GameLogic import Game, Point

custom_preamble = {
    "text.usetex": True,
    "text.latex.preamble": [
        r"\usepackage{amsmath}",  # for the align enivironment
    ],
}
plt.rcParams.update(custom_preamble)
mpl.use('TkAgg')


def print_maps(maps, fill='\u2590\u2588\u258C'):
    """
    Print one or multiple maps to console (Developer Tool)
    :param maps: (boolean) numpy array with shape [maps_count, size_x, size_y]
    :param fill: Define 3 characters for a true value (optional. default: 1/2 right block + 1 block + 1/2 left block)
    :return:
    """
    if maps.ndim == 2:
        maps = np.expand_dims(maps, 0)

    if maps.ndim != 3:
        raise ValueError('Invalid number of dimensions')

    print('\u250f' + '\u2501' * (maps.shape[2] * 3) + '\u2513')
    for i, layer in enumerate(maps):
        if i > 0:
            print('\u2523' + '\u2501' * (maps.shape[2] * 3) + '\u252B')
        for row in layer:
            print('\u2503', end='')
            for cell in row:
                if cell:
                    print(fill, end='')
                else:
                    print('   ', end='')
            print('\u2503')
    print('\u2517' + '\u2501' * (maps.shape[2] * 3) + '\u251B')


def fig_to_data(fig):
    """
    Convert a whole matplotlib figure to numpy array of pixels
    :param fig: matplotlib figure
    :return: numpy array with shape [height, width, 3] (3 for RGB)
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)

    buf = np.roll(buf, 3, axis=2)
    return buf


class Visualisation:
    def __init__(self, input_maps, map_size, agent_count, view_padding, view_reduced=False,
                 truth_obstacles=None, dt='', i_game=None, scores=None, reached=None):
        self._map_size_x = map_size[0]
        self._map_size_y = map_size[1]
        self._view_padding = view_padding
        self._view_size_x = view_padding[0] + 1 + view_padding[1] if view_reduced else 0
        self._view_size_y = view_padding[2] + 1 + view_padding[3] if view_reduced else 0
        self._view_reduced = view_reduced
        self._agent_count = agent_count
        self.time_steps = len(input_maps)
        self._next_step = False
        self._dt = datetime.now().strftime('%Y-%m-%d-%H-%M-%S') if dt == '' else dt
        self._i_game = i_game
        self._scores = scores
        self._reached = reached

        # Modify given input_map for further usage
        input_maps = np.array(input_maps)
        if not view_reduced:
            # If field of view for agents is not reduced just force input_maps to the right shape
            # and set full_maps equal to it because input maps are the full maps
            self._input_maps = np.reshape(input_maps,
                                          (self.time_steps, agent_count, self._map_size_x, self._map_size_y))
            self._full_maps = self._input_maps
        else:
            # Separate the local field of view maps and global current and aim positions
            self._input_maps = np.reshape(input_maps[:, :, 0:self._view_size_x * self._view_size_y],
                                          (self.time_steps, agent_count, self._view_size_x, self._view_size_y))
            self._current_pos = (input_maps[:, :, -4:-2] * map_size).round().astype('int64')
            self._aim_pos = (input_maps[:, :, -2:] * map_size).round().astype('int64')
            if np.unique(self._current_pos, axis=0).shape[0] > 1:
                Warning('Warning: Aim positions changed over time')

            # Create full maps of the environment and start with a empty matrix
            # The size of a single map is padded to apply agents field of view also if a agent stands close to a corner
            # Shape: (time_steps, agent_counts, X, Y)
            self._full_maps = np.zeros((self.time_steps,
                                        self._agent_count,
                                        view_padding[0] + self._map_size_x + view_padding[1],
                                        view_padding[2] + self._map_size_y + view_padding[3]))

            # Create a list of indices of all cells in full_maps to put input_maps into it at the right positions later
            # Shape: (4, time_steps * agent_counts * map_size_x * map_size_y)
            indices = np.array(list(itertools.product(np.arange(self.time_steps),
                                                      np.arange(agent_count),
                                                      np.arange(self._view_size_x),
                                                      np.arange(self._view_size_y)))).T

            # Add current agents positions as offset to indices to bring the smaller input_maps to the right positions
            indices[2] += np.repeat(self._current_pos[:, :, 0], self._view_size_x * self._view_size_y)  # x offset
            indices[3] += np.repeat(self._current_pos[:, :, 1], self._view_size_x * self._view_size_y)  # y offset

            # Fill the full maps with values from the input maps at the previously desired positions
            self._full_maps[indices[0], indices[1], indices[2], indices[3]] = self._input_maps.flatten()

            # Crop out the padding of the full maps
            self._full_maps = self._full_maps[:, :, view_padding[2]:-view_padding[3], view_padding[0]:-view_padding[1]]

        # Obstacles
        self._obstacle_maps = (self._full_maps == 0.25)
        if not np.all(np.isin(np.count_nonzero(self._obstacle_maps, axis=(0, 1)), [0, self.time_steps * agent_count])):
            Warning('Warning: Positions of obstacles changed over time or are different for different agents')
        if isinstance(truth_obstacles, type(None)):
            self._obstacle_pos = np.argwhere(np.any(self._obstacle_maps, axis=(0, 1)))
        else:
            if len(truth_obstacles) > 0:
                self._obstacle_pos = np.unique(truth_obstacles, axis=0)
            else:
                self._obstacle_pos = np.array([], dtype='int64')

        # Others Position
        self._others_maps = (self._full_maps == 0.5)

        # Aim Positions
        self._aim_maps = (self._full_maps == 0.75)
        if not np.all(np.isin(np.count_nonzero(self._aim_maps, axis=0), [0, self.time_steps])):
            Warning('Warning: Aim maps changed over time')

        # Current Positions
        self._current_maps = (self._full_maps == 1.0)
        if np.any(np.count_nonzero(self._current_maps, axis=(2, 3)) > 1):
            Warning('Warning: At least one time step there are several positions for one or more agents')
        elif np.any(np.count_nonzero(self._current_maps, axis=(2, 3)) < 1):
            Warning('Warning: At least at one time step for one or more agents the positions are missing')

        # # Agent status includes 'aim achieved' (a), 'self inflicted accident' (s), 'third-party fault accident' (3)
        # # and 'time out' (t)  # TODO: Agent status
        # self._agents_conditions = np.zeros(self._agent_count, dtype=np.dtype('U1'))

        # Color
        self._color_hue_offset = np.random.uniform()

    def get_maps_for_agent(self, time_step=-1, agent=0, plot_input=False):
        """
        Return the map for agent X's point of view including: obstacles, own aim position,
        own current position (, own next position), others current position (, others next position).
        :param time_step:
        :param agent: id number of agent
        :param plot_input:
        :return: map as boolean array with shape [4 or 6, size_x, size_y]
        """
        obstacles = self._obstacle_maps[time_step, agent]
        aim_map = self._aim_maps[time_step, agent]
        cur_map = self._current_maps[time_step, agent]
        # nxt_map = ... TODO: Next step
        others_cp = self._others_maps[time_step, agent]
        full_map = self._full_maps[time_step, agent]
        input_map = self._input_maps[time_step, agent]

        # If field of view reduced add global positions to reduced input map
        if self._view_reduced:
            # get global positions, concatenate it to a vector and add a white border above
            # -> size = [2, 4]
            c_pos = self._current_pos[time_step, agent] / [self._map_size_x, self._map_size_y]
            a_pos = self._aim_pos[time_step, agent] / [self._map_size_x, self._map_size_y]
            positions = np.concatenate([c_pos, a_pos]).reshape(1, -1)
            positions = np.concatenate([np.ones((1, positions.shape[1])), positions], axis=0)

            # duplicate each pixel in input_map and positions to force divisibility by two
            input_map = input_map.repeat(2, axis=0).repeat(2, axis=1)
            positions = positions.repeat(2, axis=0).repeat(2, axis=1)

            # align centered both
            size_diff = input_map.shape[1] - positions.shape[1]
            if size_diff < 0:  # position vector is longer than input_map width
                # add placeholder left and right to input_map
                input_map = np.concatenate([np.ones((input_map.shape[0], int(-0.5 * size_diff))),
                                            input_map,
                                            np.ones((input_map.shape[0], int(-0.5 * size_diff)))], axis=1)
            elif size_diff > 0:  # position vector is shorter than input_map width
                # add placeholder left and right to positions
                positions = np.concatenate([np.ones((4, int(0.5 * size_diff))),
                                            positions,
                                            np.ones((4, int(0.5 * size_diff)))], axis=1)

            # now concatenate both
            input_map = np.concatenate([input_map, positions], axis=0)

            # make shape of input_map quadratic
            size_diff = input_map.shape[0] - positions.shape[1]
            if size_diff < 0:  # concatenated input_map is wider than high
                input_map = np.concatenate([np.ones((int(-0.5 * size_diff), input_map.shape[1])),
                                            input_map,
                                            np.ones((int(-0.5 * size_diff), input_map.shape[1]))], axis=0)
            elif size_diff > 0:  # concatenated input_map is higher than wide
                input_map = np.concatenate([np.ones((input_map.shape[0], int(0.5 * size_diff))),
                                            input_map,
                                            np.ones((input_map.shape[0], int(0.5 * size_diff)))], axis=1)

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

    def _get_plot_color(self, agent, next_step=False):
        """
        Return color for a agents
        :param agent: id number of agent
        :param next_step:
        :return: rgb value
        """
        hue = agent / self._agent_count + self._color_hue_offset
        saturation = 1.0 if not next_step else 0.1
        value = 0.7 if not next_step else 0.9
        return colorsys.hsv_to_rgb(hue, saturation, value)

    def _plot_map_border(self, ax):
        """
        Plot a black border around a map
        :param ax: matplotlib axis / subplot
        :return:
        """
        border = patches.Rectangle((0, 0), self._map_size_y, self._map_size_x, linewidth=5, edgecolor='black',
                                   facecolor='none')
        ax.add_patch(border)

    def _plot_view_border(self, ax, pos):
        """
        Plot a grey border to visualize field of view of an agent
        :param ax: matplotlib axis / subplot
        :param pos: middle position of field (position of an agent)
        :return:
        """
        start = (pos[1] - self._view_padding[1],
                 self._map_size_x - pos[0] - 1 - self._view_padding[3])
        border = patches.Rectangle(start, self._view_size_y, self._view_size_x, linewidth=2, edgecolor='grey',
                                   facecolor='none')
        ax.add_patch(border)

    def _plot_rect_at_pos(self, ax, x, y, color):
        """
        Plot a rectangle to symbolize an agent or abstacle
        :param ax: matplotlib axis / subplot
        :param x: x position
        :param y: y position
        :param color: color of rectangle
        :return:
        """
        rect = patches.Rectangle((y, self._map_size_x - x - 1), 1, 1, linewidth=0,
                                 edgecolor='none', facecolor=color)
        ax.add_patch(rect)

    def _plot_label(self, ax, x, y, text, color):
        """
        Plot a text label
        :param ax: matplotlib axis / subplot
        :param x: x position
        :param y: y position
        :param text: text
        :param color: color of text
        :return:
        """
        x = self._map_size_x - x - 1
        prop = FontProperties(family='monospace', weight='black')
        tp = TextPath((y, x), text, prop=prop, size=1)
        polygon = tp.to_polygons()
        for a in polygon:
            patch = patches.Polygon(a, facecolor=color, edgecolor='black', linewidth=1, zorder=10)
            ax.add_patch(patch)

    def _plot_map(self, ax, map, color, plot_view_filed=False, curr_pos=None):
        """
        Plot a boolean map
        :param ax: matplotlib axis / subplot
        :param map: boolean numpy map
        :param color: color of true blocks in boolean map
        :param plot_view_filed: if true, a grey border shows the field of view of an agent
        :param curr_pos: position of agent to plot view field border at the right position
        :return:
        """
        # Plot map
        for x in range(self._map_size_x):
            for y in range(self._map_size_y):
                if map[x, y]:
                    self._plot_rect_at_pos(ax, x, y, color)

        # Plot view field
        if plot_view_filed:
            self._plot_view_border(ax, curr_pos)

        # Plot border
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
        """
        Plot a heatmap
        :param ax: matplotlib axis / subplot
        :param map: map to plot
        :return:
        """
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

    def _plot_info(self, ax, time_step):
        """
        Plot a text field with information about number of current game, time step, sizes of map of field of view
        :param ax: matplotlib axis / subplot
        :param time_step: number of time step
        :return:
        """
        text = r'\begin{align*}'
        if not isinstance(self._i_game, type(None)):
            text += r'i_{{game}}&={}\\'.format(self._i_game)
        text += r't&={}\\'.format(time_step)
        text += r'size_{{map}}&=\left[{}\times{}\right]\\'.format(self._map_size_x, self._map_size_y)
        if self._view_reduced:
            text += r'size_{{view}}&=\left[{}\times{}\right]\\'.format(self._view_size_x, self._view_size_y)
        text += r'\end{align*}'

        ax.text(0.3, 1.0, text, fontsize=17, ha='left', va='top')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    def _plot_overview(self, fig, outer_grid=None, time_step=-1, plot_agent_status=True, plot_path=True,
                       plot_input=False, plot_info=False, title=''):
        """
        Plot a map all agents are included
        :param fig: matplotlib figure
        :param outer_grid: none or matplotlib grid this map should be plotted in
        :param time_step: time step to be plotted
        :param plot_agent_status: plot agents status (not implemented yet)
        :param plot_path: plot a line from start via each step to the current position
        :param plot_input: unused parameter. Just here to make list of parameters equal to _plot_all()
        :param plot_info: plot some information below the map
        :param title: title of plot
        :return: matplotlib figure
        """
        if outer_grid is None:
            outer_grid = gridspec.GridSpec(1, 1, wspace=0, hspace=0)[0]
            grid = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_grid,
                                                    wspace=0.1, hspace=0.1, width_ratios=[1],
                                                    height_ratios=[0, 5, 1 if plot_info else 0])
        else:
            grid = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_grid,
                                                    wspace=0.1, hspace=0.1, width_ratios=[1],
                                                    height_ratios=[1 if plot_info else 0, 3, 1 if plot_info else 0])
        ax = plt.Subplot(fig, grid[1])

        # Obstacles
        for x, y in self._obstacle_pos:
            self._plot_rect_at_pos(ax, x, y, 'black')

        # Add maps of all agents to the plot
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
            # color_next = self._get_plot_color(i_agent, next_step=True)  # TODO: Next step

            # Plot next position
            # TODO: Next step
            # if self._next_step:
            #     self._plot_map(ax, nxt_map, color_next)

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
            for x, y in start_pos:
                self._plot_label(ax, x - 0.15, y + 0.2, "S", color)

            # Plot aim position
            for x, y in aim_pos:
                self._plot_label(ax, x - 0.15, y + 0.2, "E", color)

            # # Plot agent status  # TODO: Agent status
            # if plot_agent_status:
            #     for status, symbol in zip(['a', 's', '3', 't'], ['\u2713', '\u2717', '\u2717', '\u2717']):  # \u2620
            #         if self._agents_conditions[i_agent] == status:
            #             for x, y in self._current_maps[time_step, i_agent]:
            #                 self._plot_label(ax, x - 0.15, y + 0.2, symbol, 'black')

        # Plot Border
        self._plot_map_border(ax)

        ax.set_ylim(0, self._map_size_x)
        ax.set_xlim(0, self._map_size_y)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        ax.set_title(title, fontsize=15)
        fig.add_subplot(ax)

        # Add info box below if wanted
        if plot_info:
            ax = plt.Subplot(fig, grid[2])
            self._plot_info(ax, time_step)
            fig.add_subplot(ax)

        return fig

    def _plot_all(self, fig, time_step=-1, plot_agent_status=True, plot_path=True, plot_input=False,
                  plot_info=False, overview_title='Overview'):
        """
        Plot a visualisation with a big overview map and
        small maps for all agents for all types of object in the environment
        :param fig: matplotlib figure
        :param time_step: time step to be plotted
        :param plot_agent_status: plot agents status (not implemented yet)
        :param plot_path: plot a line from start via each step to the current position
        :param plot_input: show also heatmaps to visualize network input
        :param plot_info: plot some information below the overview map
        :param overview_title: title of overview map
        :return: matplotlib figure
        """
        # Create outer grid
        outer = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1, width_ratios=[0.382, 0.618])
        outer.update(left=0.01, right=0.99, top=0.95, bottom=0.01)

        # Plot overview (and information box) on the left side
        self._plot_overview(fig, outer[0], time_step=time_step, plot_agent_status=plot_agent_status,
                            plot_path=plot_path, plot_info=plot_info, title=overview_title)

        # Define titles of the small maps
        if self._next_step:
            nr_maps = 6
            maps_names = ['Obstacles', 'Aim', 'Agent\'s\nCurrent Pos.', 'Agent\'s\nNext Pos.',
                          'Others\nCurrent Pos.', 'Others\nNext Pos.']
        else:
            nr_maps = 4
            maps_names = ['Obstacles', 'Aim', 'Agent\'s\nPosition', 'Others\nPosition']
        if plot_input:
            nr_maps += 2
            maps_names.append('Full Map\nNet Input')
            maps_names.append('Reduced\nNet Input')

        # Create the right grid for the small maps
        agents_grid = gridspec.GridSpecFromSubplotSpec(self._agent_count, nr_maps, subplot_spec=outer[1],
                                                       wspace=0.1, hspace=0.1)

        # Plot small maps
        for i_agent in range(self._agent_count):
            maps = self.get_maps_for_agent(time_step=time_step, agent=i_agent, plot_input=plot_input)
            for i_map, map_ in enumerate(maps):
                i_grid = i_agent * nr_maps + i_map
                ax = plt.Subplot(fig, agents_grid[i_grid])
                if plot_input and i_map + 2 >= nr_maps:
                    self._plot_heatmap(ax, map_)
                else:
                    color = self._get_plot_color(i_agent)
                    if self._view_reduced:
                        self._plot_map(ax, map_, color, plot_view_filed=True,
                                       curr_pos=self._current_pos[time_step, i_agent])
                    else:
                        self._plot_map(ax, map_, color)

                # add map titles
                if ax.is_first_row():
                    ax.set_xlabel(maps_names[i_map], fontsize=15)
                    ax.xaxis.set_label_position('top')

                # add agent titles
                if ax.is_first_col():
                    ax.set_ylabel('Agent {}'.format(i_agent), fontsize=15)

                fig.add_subplot(ax)

        plt.subplots_adjust(wspace=0, hspace=0)

        return fig

    def plot_map(self, map_, block=True, save_as=None):
        # Its a wrapper method of _plot_map()
        """
        Shows a boolean map
        :param map_: number of agent
        :param block: blocking behavior of plt.show(block=...)
        :param save_as: string of path if plot should be saved instead of displayed
        :return:
        """
        # Disable tools and create figure and axes
        mpl.rcParams['toolbar'] = 'None'
        fig, ax = plt.subplots(1, figsize=(5, 5))

        # Plot map
        self._plot_map(ax, map_, 'black')

        # Save of show plot
        if save_as:
            fig.savefig(save_as)
            plt.close(fig)
        else:
            plt.show(block=block)

    def plot_overview(self, time_step=-1, plot_agent_status=True, plot_path=True, plot_info=False,
                      block=True, save=False):
        # Its a wrapper method of _plot_overview()
        """
        Plot a map all agents are included
        :param time_step: time step to be plotted
        :param plot_agent_status: plot agents status (not implemented yet)
        :param plot_path: plot a line from start via each step to the current position
        :param plot_info: plot some information below the map
        :param block: blocking behavior of plt.show(block=...)
        :param save: if true save plot at viz/... instead of displaying
        :return:
        """
        if time_step == -1:
            time_step = self.time_steps - 1

        # Disable tools and create figure and axes
        mpl.rcParams['toolbar'] = 'None'
        img_width = 1080
        img_height = 1080
        dpi = 120
        fig = plt.figure(figsize=(img_width / dpi, img_height / dpi), dpi=dpi)

        # Plot overview
        self._plot_overview(fig, time_step=time_step, plot_agent_status=plot_agent_status, plot_path=plot_path,
                            plot_info=plot_info)
        fig.set_size_inches(img_width / dpi, img_height / dpi)

        # Save of show plot
        if save:
            directory = os.path.join('viz', self._dt, 'overview')

            # Check if directory for images exists
            if not os.path.exists(directory):
                os.makedirs(directory)

            file_name = f'{self._dt}_game_{self._i_game}_time_{time_step}_overview.png'
            fig.savefig(os.path.join(directory, file_name), dpi=dpi)
            plt.close(fig)  # leads to a crash Python, if this method is executed too often in short time :-(
        else:
            plt.show(block=block)

    def plot_all(self, time_step=-1, plot_agent_status=True, plot_path=True, plot_input=False, plot_info=False,
                 block=True, save=False):
        # Its a wrapper method of _plot_all()
        """
        Plot a visualisation with a big overview map and
        small maps for all agents for all types of object in the environment
        :param time_step: time step to be plotted
        :param plot_agent_status: plot agents status (not implemented yet)
        :param plot_path: plot a line from start via each step to the current position
        :param plot_input: show also heatmaps to visualize network input
        :param plot_info: plot some information below the overview map
        :param block: blocking behavior of plt.show(block=...)
        :param save: if true save plot at viz/... instead of displaying
        :return:
        """
        if time_step == -1:
            time_step = self.time_steps - 1

        # Disable tools and create figure, axes and outer grid
        mpl.rcParams['toolbar'] = 'None'
        img_width = 1920
        img_height = 1080
        dpi = 120
        fig = plt.figure(figsize=(img_width / dpi, img_height / dpi), dpi=dpi)

        # Plot all
        fig = self._plot_all(fig, time_step=time_step, plot_agent_status=plot_agent_status,
                             plot_path=plot_path, plot_input=plot_input, plot_info=plot_info)
        fig.set_size_inches(img_width / dpi, img_height / dpi)

        # Save of show plot
        if save:
            directory = os.path.join('viz', self._dt, 'all')

            # Check if directory for images exists
            if not os.path.exists(directory):
                os.makedirs(directory)

            file_name = f'{self._dt}_game_{self._i_game}_time_{time_step}_all.png'
            fig.savefig(os.path.join(directory, file_name), dpi=dpi)
            plt.close(fig)  # leads to a crash Python, if this method is executed too often in short time :-(
        else:
            plt.show(block=block)

    def generate_mp4(self, kind, plot_agent_status=True, plot_path=True, plot_input=False, plot_info=True):
        """
        Generate and save a mp4 video from desired kind of plot over all time steps
        :param kind: kind of plot as string ('all' or 'overview')
        :param plot_agent_status: plot agents status (not implemented yet)
        :param plot_path: plot a line from start via each step to the current position
        :param plot_input: show also heatmaps to visualize network input
        :param plot_info: plot some information below the overview map
        """
        plot_func = None
        img_width = 1920
        img_height = 1080
        dpi = 120

        # Get right plot function depending of desired kind of plot
        if kind == 'all':
            plot_func = self._plot_all
        elif kind == 'overview':
            plot_func = self._plot_overview
            img_height = 1080
            img_width = 1080  # 900

        plt.ioff()  # prevent matplotlib from running out of memory

        def draw_frame(ts):
            fig = plt.figure(figsize=(img_width / dpi, img_height / dpi), dpi=dpi)
            fig = plot_func(fig, time_step=ts, plot_agent_status=plot_agent_status, plot_path=plot_path,
                            plot_input=plot_input, plot_info=plot_info)
            fig.set_size_inches(img_width / dpi, img_height / dpi)
            data = fig_to_data(fig)
            # fig.clf()
            # plt.clf()
            plt.close(fig)
            return data

        # # Make the pool of workers  # TODO: multithreading
        # pool = mp.ProcessingPool(mp.cpu_count() - 1)
        # # Start multithreading
        # frame_array = list(tqdm(pool.imap(draw_frame, np.arange(self.time_steps)), total=self.time_steps))
        frame_array = [draw_frame(ts) for ts in tqdm(range(self.time_steps))]

        # # Close the pool and wait for the work to finish  # TODO: multithreading
        # pool.close()
        # pool.join()

        directory = os.path.join('viz', self._dt, kind)

        # Check if directory for images exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_name = f'{self._dt}_game_{self._i_game}_{kind}.mp4'

        # Generate video
        w = imageio.get_writer(os.path.join(directory, file_name),
                               fps=4, quality=6, macro_block_size=20)
        for i in range(len(frame_array)):
            w.append_data(frame_array[i])
        w.close()

        # Add entry to a text file including all videos of a run
        # You can use ffmpeg to concatenate videos of a run:
        # ffmpeg -f concat -safe 0 -i mylist.txt -c copy output.mp4
        # https://stackoverflow.com/a/11175851/7439335
        open(os.path.join(directory, 'videos.txt'), "a").write(f"file '{file_name}'\n")

    def save(self):
        """
        Save Visualisation object at viz/...
        :return:
        """
        directory = os.path.join('viz', self._dt, 'obj')

        # Check if directory for viz exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save
        f = open(os.path.join(directory, f'{self._dt}_game_{self._i_game}.viz'), 'wb')
        pickle.dump(self, f, 2)
        f.close()

    @staticmethod
    def load(path):
        """
        Load Visualisation object from file
        :param path: path of saved Visualisation object
        :return:
        """
        f = open(path, 'rb')
        viz = pickle.load(f)
        f.close()
        return viz


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Environment')
    parser.add_argument("-f", "--file_path", type=str,
                        help="define path to .viz file")
    args = parser.parse_args()

    viz = Visualisation.load(args.file_path)
    viz._color_hue_offset = 0.3
    viz.generate_mp4('all', plot_input=True, plot_info=False)
