import copy
import colorsys
import warnings
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties


class Map:
    def __init__(self, size_x, size_y, agent_count=1, obstacle_map=None, next_step=False, load_from_file=None):
        if load_from_file:
            f = open(load_from_file, 'rb')
            tmp_dict = pickle.load(f)
            f.close()
            self.__dict__.update(tmp_dict)
        else:
            self._size_x = size_x
            self._size_y = size_y
            self._agent_count = agent_count
            self._next_step = next_step

            # Check input values
            if agent_count < 1:
                raise ValueError('number of agents must be at least one')
            if size_x * size_y < 2 * agent_count:
                raise ValueError('board size is to small')

            # Create layer array of type char to filter the map
            self._layers = np.array([0, 'o'], dtype=np.dtype('U1'))  # obstacles
            for p in range(agent_count):
                unstacked_layers = [self._layers,
                                    np.array([p + 1, 'a'], dtype=np.dtype('U1')),  # aim position
                                    np.array([p + 1, 'c'], dtype=np.dtype('U1'))]  # current position
                if next_step:
                    unstacked_layers.append(np.array([p + 1, 'n'], dtype=np.dtype('U1')))  # next position
                self._layers = np.vstack(unstacked_layers)  # shape: (obstacle+agent_count*(aim+current[+ next]), 2)
                # -> 2 for two filters: agents and layers

            # Create map skeleton
            nr_layers = 1 + agent_count * (3 if next_step else 2)  # obstacle + agent_count * (aim + current [+ next])
            self._map = np.zeros((nr_layers, size_x, size_y), dtype=bool)  # shape: (nr_layers, size_x, size_y)

            # Add obstacles to map
            if obstacle_map is not None:
                if not isinstance(obstacle_map, np.ndarray):
                    raise TypeError('obstacle map must ba a numpy array')
                if obstacle_map.shape == (size_x, size_y):
                    self._map[0, :, :] = obstacle_map
                else:
                    raise ValueError('obstacle map must have the same size as main map')

            # History
            self._hist = np.zeros((0, agent_count, 2))  # shape: (time steps, agent_count, 2) -> 2 for x & y
            self._hist_next = np.zeros((0, agent_count, 2))  # shape: (time steps, agent_count, 2) -> 2 for x & y

            # Agent status includes 'aim achieved' (a), 'self inflicted accident' (s) 'third-party fault accident' (3)
            self._agent_status = np.zeros(agent_count, dtype=np.dtype('U1'))

            # Color
            self._color_hue_offset = np.random.uniform()

    def _layer_filter(self, agent=None, layer=None):
        agent = None if agent is None else str(int(agent) + 1)  # in self._layers agent number starts at one
        agent_filter = [True for _ in range(self._layers.shape[0])] if agent is None else self._layers[:, 0] == agent
        layer_filter = [True for _ in range(self._layers.shape[0])] if layer is None else self._layers[:, 1] == layer
        return np.array(agent_filter) & np.array(layer_filter)

    def _generate_layers_from_positions(self, positions):
        positions = np.concatenate([np.expand_dims(np.arange(self._agent_count), 1), positions], 1)
        layers = np.zeros((self._agent_count, self._size_x, self._size_y), dtype=bool)
        layers[positions[:, 0], positions[:, 1], positions[:, 2]] = [True for _ in range(self._agent_count)]
        return layers

    def _add_hist(self, positions, next_step=False):
        if next_step:
            self._hist_next = np.concatenate([self._hist, positions], 0)
        else:
            self._hist = np.concatenate([self._hist, positions], 0)

    def set_position(self, agent, layer, x, y):
        """
        Define a position (current, aim, next) for a given agent.
        :param agent: number of agent
        :param layer: 'a' for aim position, 'c' for current position or 'n' for next position
        :param x: x coordinate of new position
        :param y: y coordinate of new position
        """
        ary = np.zeros((self._size_x, self._size_y))
        ary[x, y] = True
        self._map[self._layer_filter(agent=agent, layer=layer)] = ary

        if layer == 'c':
            warnings.warn("This method does not update the history and agent status! Use move_agents() instead.")

    def _set_positions(self, layer, positions):
        """
        Define [aim, current, next] positions for all agents
        :param layer: 'a' for aim positions, 'c' for current positions, 'n' for next positions or None for all
        :param positions: (boolean) numpy array with shape (agent_count, 2) and x & y coordinates inside for all agents.
        In case of layer=None shape of positions must be (obstacle + agent_count * (aim + current [+ next]), 2)
        """
        if layer == 'n' and not self._next_step:
            raise Exception('Next position not activated in this map')

        # Add new positions to history
        if layer == 'c':
            self._add_hist(np.expand_dims(positions, 0))
        if layer == 'n' and self._next_step:
            self._add_hist(np.expand_dims(positions, 0), next_step=True)

        self._map[self._layer_filter(layer=layer)] = self._generate_layers_from_positions(positions)

    def set_aim_positions(self, positions):
        """
        Define aim positions for all agents
        :param positions: (boolean) numpy array with shape (agent_count, 2) and x & y coordinates inside for all agents
        """
        self._set_positions(layer='a', positions=positions)

    def set_current_positions(self, positions):
        """
        Define current positions for all agents
        :param positions: (boolean) numpy array with shape (agent_count, 2) and x & y coordinates inside for all agents
        """

        if self._hist.shape[0] > 0:
            warnings.warn("This method does not update agent status! Use move_agents() instead.")

        self._set_positions(layer='c', positions=positions)

    def set_next_positions(self, positions):
        """
        Define next positions for all agents
        :param positions: (boolean) numpy array with shape (agent_count, 2) and x & y coordinates inside for all agents
        """
        self._set_positions(layer='n', positions=positions)

    def get_map(self):
        """
        Returns the whole map with all layers
        :return: map as boolean array with shape (nr_layers, size_x, size_y)
        """
        return copy.deepcopy(self._map)

    def get_filtered_map(self, agent=None, layer=None):
        """
        Returns a filtered map with only wanted layers
        :param agent: number of agent, None for all agents
        :param layer: 'o' for obstacles, 'a' for aim positions, 'c' for current positions,
        'n' for next positions, None for all layers
        :return: filtered map as boolean array with shape (requested layers, size_x, size_y)
        """
        return copy.deepcopy(self._map[self._layer_filter(agent, layer)])

    def get_map_for_agent(self, agent, view_filed=None):
        """
        Get map for agent X's point of view
        :param agent: number of agent
        :param view_filed: size of view field
        :return: map as boolean array with shape (, size_x, size_y)
        """
        obstacles = self._map[self._layer_filter(layer='o')]
        a_c_n_pos = self._map[self._layer_filter(agent=agent)]
        others_cp = np.any(self._map[self._layer_filter(layer='c')], axis=0)  # current positions of other agents
        others_cp = others_cp & ~self._map[self._layer_filter(agent=agent, layer='c')]  # subtract own current position
        # TODO: field of view
        # others_cp = others_cp & ...
        agent_map = np.concatenate((obstacles, a_c_n_pos, others_cp))

        if self._next_step:
            others_np = np.any(self._map[self._layer_filter(layer='n')], axis=0)  # next positions of other agents
            others_np = others_np & ~self._map[self._layer_filter(agent=agent, layer='n')]  # subtract own next position
            # TODO: field of view
            # others_np = others_np & ...
            agent_map = np.concatenate((agent_map, others_np))

        return agent_map

    def get_positions(self, agent=None, layer=None):
        """
        Returns positions as coordinates of wanted agents and layers
        :param agent:
        :param layer:
        :return:
        """
        return np.argwhere(self.get_filtered_map(agent=agent, layer=layer))[:, 1:3]

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

    def get_agent_status(self):
        return self._agent_status

    def move_agents(self, commands):
        """
        Checks if desired commands of agents lead to accidents and
        updates the map accordingly for agents without an accident.
        :param commands: (boolean) numpy array with shape (agent_count, 5) which contains one hot vectors for all agents
        One hot vectors contain: [stay, up, right, down, left]
        :return: boolean array with shape (agent_count, 1) with True if the command is valid for this agent and
        False if there was an accident.
        """

        old_pos_maps = self.get_filtered_map(layer='c')
        old_pos_coord = self.get_positions(layer='c')

        # Calculate desired positions
        offset_kernel = [[0, 0],
                         [-1, 0],
                         [0, 1],
                         [1, 0],
                         [0, -1]]
        offset = np.matmul(commands, offset_kernel)
        desired_pos_coord = old_pos_coord + offset

        # If an agent has reached its destination or has had an accident, it is not allowed to move on
        allowed_pos_coord = np.where(np.expand_dims(np.isin(self._agent_status, ['a', 's', '3']), axis=1),
                                     old_pos_coord, desired_pos_coord)

        # Check whether an agent has already left the map
        accident = np.any(np.concatenate([allowed_pos_coord < 0,
                                          allowed_pos_coord >= [self._size_x, self._size_y]], axis=1), axis=1)

        # If there is an accident for this agent the position stays the old (and he dies there)
        allowed_pos_coord = np.where(np.expand_dims(accident, axis=1), old_pos_coord, allowed_pos_coord)

        # Generate maps for all agents with their desired positions
        desired_pos_maps = self._generate_layers_from_positions(allowed_pos_coord)

        # Create array contains all positions which creates an accident with other agents
        # -> shape: (size_x, size_y)
        # Its contains: old positions, desired positions
        danger_zones = np.any([np.any(old_pos_maps, axis=0),
                               np.any(desired_pos_maps, axis=0)], axis=0)

        # Make danger zones more specific for all single agent (remove own positions)
        # -> shape (agent_count, size_x, size_y)
        danger_zones = np.all([np.tile(danger_zones, (self._agent_count, 1, 1)),
                               ~old_pos_maps,
                               ~desired_pos_maps], axis=0)

        # Add obstacles to danger zones
        danger_zones = np.any([np.tile(np.squeeze(self.get_filtered_map(layer='o')), (self._agent_count, 1, 1)),
                               danger_zones], axis=0)

        # Check whether an agent crash into an obstacle or other agent
        # print('Danger Zones:')
        # self.print_layers(danger_zones)
        accident = np.any([accident, danger_zones[np.arange(self._agent_count),
                                                  allowed_pos_coord[:, 0],
                                                  allowed_pos_coord[:, 1]]], axis=0)

        # If there is an accident for this agent the position stays the old (and he dies there)
        allowed_pos_coord = np.where(np.expand_dims(accident, axis=1), old_pos_coord, allowed_pos_coord)

        # TODO: third-party fault accident

        # Update agents position
        self._set_positions(layer='c', positions=allowed_pos_coord)  # TODO: Next Step

        # Update agents status
        # 'aim achieved' (a),
        goal_achieved = self.get_filtered_map(layer='a')[np.arange(self._agent_count),
                                                         allowed_pos_coord[:, 0],
                                                         allowed_pos_coord[:, 1]]
        self._agent_status = np.where(accident, 's', self._agent_status)  # 'self inflicted accident' (s)
        # TODO: # 'third-party fault accident' (3)
        self._agent_status = np.where(goal_achieved, 'a', self._agent_status)

    def print_layers(self, layers, fill='\u2590\u2588\u258C'):
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

    def _plot_label(self, ax, x, y, text, color):
        prop = FontProperties(family='monospace', weight='black')
        tp = TextPath((x, y), text, prop=prop, size=1)
        polygon = tp.to_polygons()
        for a in polygon:
            patch = patches.Polygon(a, facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(patch)

    def _plot_border(self, ax, size_x, size_y):
        border = patches.Rectangle((0, 0), size_y, size_x, linewidth=5, edgecolor='black',
                                   facecolor='none')
        ax.add_patch(border)

    def _get_plot_color(self, agent_index, next_step=False):
        hue = 1.0 / self._agent_count * agent_index + self._color_hue_offset
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
            a_c_n_map = self._map[self._layer_filter(agent=i_agent)]

            color = self._get_plot_color(i_agent, next_step=False)
            color_next = self._get_plot_color(i_agent, next_step=True)

            # Plot next position
            if self._next_step:
                self._plot_layer(ax, a_c_n_map[2], color_next)

            # Plot current position
            self._plot_layer(ax, a_c_n_map[1], color)

            # Plot path
            if plot_path:
                offset = (1/(self._agent_count + 1) * (i_agent + 1) * 0.5) - 0.25
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
                for status, symbol in zip(['a', 's', '3'], ['\u2713', '\u2717', '\u2717']):  # \u2620
                    if self._agent_status[i_agent] == status:
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

    def save(self, path):
        f = open(path, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()


if __name__ == '__main__':
    o = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]])
    arena = Map(o.shape[0], o.shape[1], 4, o, next_step=False)

    arena.set_aim_positions([[0, o.shape[1] - 1],
                             [1, o.shape[1] - 1],
                             [2, o.shape[1] - 1],
                             [o.shape[0] - 1, 0]])
    arena.set_current_positions([[0, 0],
                                 [1, 0],
                                 [2, 0],
                                 [0, 4]])
    arena.move_agents([[0, 0, 1, 0, 0],
                       [1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 1, 0]])
    print(arena.get_agent_status())
    # arena.set_next_positions([[0, 2],
    #                           [1, 2],
    #                           [3, 1],
    #                           [2, 4]])
    # print(arena.get_map())
    # print(arena.get_map_for_agent(0))
    # print(game_map.get_filtered_map(agent='2', layer='a'))

    # arena.plot_overview()
    # arena.plot_layer(arena.get_map_for_agent(0)[0])
    arena.plot_all()
