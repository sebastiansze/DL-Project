import copy
import colorsys
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties


class Map:
    def __init__(self, size_x, size_y, agent_count=1, obstacle_map=None, next_step=False):
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

    def _layer_filter(self, agent=None, layer=None):
        agent = None if agent is None else str(int(agent) + 1)  # in self._layers agent number starts at one
        agent_filter = [True for _ in range(self._layers.shape[0])] if agent is None else self._layers[:, 0] == agent
        layer_filter = [True for _ in range(self._layers.shape[0])] if layer is None else self._layers[:, 1] == layer
        return np.array(agent_filter) & np.array(layer_filter)

    def _add_hist(self, positions):
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
            warnings.warn("This method does not update the history")

    def set_positions(self, layer, positions):
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
            self._hist = np.concatenate([self._hist, np.expand_dims(positions, 0)])

        positions = np.concatenate([np.expand_dims(np.arange(self._agent_count), 1), positions], 1)
        ary = np.zeros((self._agent_count, self._size_x, self._size_y))
        ary[positions[:, 0], positions[:, 1], positions[:, 2]] = [True for _ in range(self._agent_count)]
        self._map[self._layer_filter(layer=layer)] = ary

    def set_aim_positions(self, positions):
        """
        Define aim positions for all agents
        :param positions: (boolean) numpy array with shape (agent_count, 2) and x & y coordinates inside for all agents
        """
        self.set_positions(layer='a', positions=positions)

    def set_current_positions(self, positions):
        """
        Define current positions for all agents
        :param positions: (boolean) numpy array with shape (agent_count, 2) and x & y coordinates inside for all agents
        """
        self.set_positions(layer='c', positions=positions)

    def set_next_positions(self, positions):
        """
        Define next positions for all agents
        :param positions: (boolean) numpy array with shape (agent_count, 2) and x & y coordinates inside for all agents
        """
        self.set_positions(layer='n', positions=positions)

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

    def move_agents(self, offset_x, offset_y):
        old_positions = np.where(self.get_filtered_map(layer='c'))
        print(old_positions)
        # TODO: implement
        pass

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

    def _draw_label(self, ax, x, y, text, color):
        prop = FontProperties(family='monospace', weight='black')
        tp = TextPath((x, y), text, prop=prop, size=1)
        polygon = tp.to_polygons()
        for a in polygon:
            patch = patches.Polygon(a, facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(patch)

    def _draw_border(self, ax, size_x, size_y):
        border = patches.Rectangle((0, 0), size_y, size_x, linewidth=5, edgecolor='black',
                                   facecolor='none')
        ax.add_patch(border)

    def _get_draw_color(self, agent_index, next_step=False):
        hue = 1.0 / self._agent_count * agent_index
        saturation = 1.0 if not next_step else 0.1
        value = 0.7 if not next_step else 0.9
        return colorsys.hsv_to_rgb(hue, saturation, value)

    def _draw_layer(self, ax, layer, color):
        # Draw Layer
        for x in range(self._size_x):
            for y in range(self._size_y):
                if layer[x, y]:
                    rect = patches.Rectangle((y, self._size_x - x - 1), 1, 1, linewidth=0,
                                             edgecolor='none', facecolor=color)
                    ax.add_patch(rect)

        # Draw Border
        self._draw_border(ax, self._size_x, self._size_y)

        ax.set_ylim(0, self._size_x)
        ax.set_xlim(0, self._size_y)
        plt.axis('off')

    def _draw_overview(self, ax):
        # Obstacles
        obstacles = self.get_filtered_map(layer='o')[0]
        self._draw_layer(ax, obstacles, 'black')

        # Agents fields
        for i_agent in range(self._agent_count):
            start_pos = self.get_start_positions(agent=i_agent)
            a_c_n_pos = self._map[self._layer_filter(agent=i_agent)]

            color = self._get_draw_color(i_agent, next_step=False)
            color_next = self._get_draw_color(i_agent, next_step=True)

            # Plot next position
            if self._next_step:
                self._draw_layer(ax, a_c_n_pos[2], color_next)

            # Plot current position
            self._draw_layer(ax, a_c_n_pos[1], color)

            # Plot start position
            if start_pos is not None:
                x = start_pos[1] + 0.2
                y = self._size_y - start_pos[0] - 1 + 0.15
                self._draw_label(ax, x, y, "S", color)

            # Plot aim position
            for y, x in zip(*np.where(a_c_n_pos[0])):
                x = x + 0.2
                y = self._size_y - y - 1 + 0.15
                self._draw_label(ax, x, y, "E", color)

        # Draw Border
        self._draw_border(ax, self._size_x, self._size_y)

        ax.set_ylim(0, self._size_x)
        ax.set_xlim(0, self._size_y)
        ax.axis('off')

    def plot_layer(self, layer):
        # Create figure and axes
        fig, ax = plt.subplots(1, figsize=(5, 5))

        # Draw Layer
        self._draw_layer(ax, layer, 'black')

        plt.show()

    def plot_overview(self):
        """
        Use Matplotlib to show an overview for humans
        :return:
        """
        # Create figure and axes
        fig, ax = plt.subplots(1, figsize=(5, 5))

        # Draw Overview
        self._draw_overview(ax)

        plt.show()

    def plot_all(self):
        fig = plt.figure()
        outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)

        # Plot Overview
        ax = plt.Subplot(fig, outer[0])
        self._draw_overview(ax)
        fig.add_subplot(ax)

        # Plot Layers
        nr_agents = self._agent_count
        nr_layers = 6 if self._next_step else 5
        inner = gridspec.GridSpecFromSubplotSpec(nr_agents, nr_layers, subplot_spec=outer[1], wspace=0.1, hspace=0.1)
        for i_agent in range(nr_agents):
            layers = arena.get_map_for_agent(i_agent)
            for i_layer, layer in enumerate(layers):
                i_grid = i_agent * nr_layers + i_layer
                ax = plt.Subplot(fig, inner[i_grid])
                color = self._get_draw_color(i_agent)
                self._draw_layer(ax, layer, color)
                fig.add_subplot(ax)

        plt.show()


if __name__ == '__main__':
    o = np.array([[0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])
    arena = Map(6, 6, 3, o, next_step=True)

    arena.set_aim_positions([[0, 5],
                             [1, 5],
                             [2, 5    ]])
    arena.set_current_positions([[0, 0],
                                 [1, 0],
                                 [2, 0]])
    arena.set_current_positions([[0, 1],
                                 [1, 1],
                                 [2, 1]])
    arena.set_next_positions([[0, 2],
                              [1, 2],
                              [2, 2]])
    print(arena.get_map())
    print(arena.get_map_for_agent(0))
    # print(game_map.get_filtered_map(agent='2', layer='a'))

    # arena.plot_overview()
    # arena.plot_layer(arena.get_map_for_agent(0)[0])
    arena.plot_all()
