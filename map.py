import copy
import colorsys
import numpy as np
import matplotlib.pyplot as plt
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
            self._layers = np.vstack(unstacked_layers)

        # Create map skeleton
        nr_layers = 1 + agent_count * (3 if next_step else 2)
        self._map = np.zeros((nr_layers, size_x, size_y), dtype=bool)

        # Add obstacles to map
        if obstacle_map is not None:
            if not isinstance(obstacle_map, np.ndarray):
                raise TypeError('obstacle map must ba a numpy array')
            if obstacle_map.shape == (size_x, size_y):
                self._map[0, :, :] = obstacle_map
            else:
                raise ValueError('obstacle map must have the same size as main map')

        self._hist = np.zeros((0, agent_count, 2))

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

    def set_positions(self, layer, positions):
        """
        Define [aim, current, next] positions for all agents
        :param layer: 'a' for aim positions, 'c' for current positions or 'n' for next positions, None for all
        :param positions: (boolean) numpy array with shape [agent_count * 2] and x & y coordinates inside for all agents
        """
        if layer == 'n' and not self._next_step:
            raise Exception('Next position not activated in this map')

        # add to history
        if layer == 'c':
            self._hist = np.concatenate([self._hist, np.expand_dims(positions, 0)])

        positions = np.concatenate([np.expand_dims(np.arange(self._agent_count), 1), positions], 1)
        ary = np.zeros((self._agent_count, self._size_x, self._size_y))
        ary[positions[:, 0], positions[:, 1], positions[:, 2]] = [True for _ in range(self._agent_count)]
        self._map[self._layer_filter(layer=layer)] = ary

    def set_aim_positions(self, positions):
        """
        Define aim positions for all agents
        :param positions: (boolean) numpy array with shape [agent_count * 2] and x & y coordinates inside for all agents
        """
        self.set_positions(layer='a', positions=positions)

    def set_current_positions(self, positions):
        """
        Define current positions for all agents
        :param positions: (boolean) numpy array with shape [agent_count * 2] and x & y coordinates inside for all agents
        """
        self.set_positions(layer='c', positions=positions)

    def set_next_positions(self, positions):
        """
        Define next positions for all agents
        :param positions: (boolean) numpy array with shape [agent_count * 2] and x & y coordinates inside for all agents
        """
        self.set_positions(layer='n', positions=positions)

    def get_map(self):
        return copy.deepcopy(self._map)

    def get_filtered_map(self, agent=None, layer=None):
        return copy.deepcopy(self._map[self._layer_filter(agent, layer)])

    def get_map_for_agent(self, agent, view_filed=None):
        obstacles = self._map[self._layer_filter(layer='o')]
        a_c_n_pos = self._map[self._layer_filter(agent=agent)]
        other_pos = np.any(self._map[self._layer_filter(layer='c')], axis=0)
        other_pos = other_pos & ~self._map[self._layer_filter(agent=agent, layer='c')]  # subtract own current position
        # TODO: field of view
        # other_pos = other_pos & ...
        return np.concatenate((obstacles, a_c_n_pos, other_pos))

    def get_start_positions(self, agent=None):
        if self._hist.shape[0] == 0:
            return None
        if agent is None:
            return self._hist[0]
        else:
            return self._hist[0, agent]

    def plot_overview(self):
        def draw_layer(ax, layer, color):
            for x in range(self._size_x):
                for y in range(self._size_y):
                    if layer[x, y]:
                        rect = patches.Rectangle((y, self._size_x - x - 1), 1, 1, linewidth=0, edgecolor='none',
                                                 facecolor=color)
                        ax.add_patch(rect)

        def draw_label(ax, x, y, text, color):
            prop = FontProperties(family='monospace', weight='black')
            tp = TextPath((x, y), text, prop=prop, size=1)
            polygon = tp.to_polygons()
            for a in polygon:
                patch = patches.Polygon(a, facecolor=color, edgecolor='black', linewidth=1)
                ax.add_patch(patch)

        # Create figure and axes
        fig, ax = plt.subplots(1, figsize=(5, 5))

        # Obstacles
        obstacles = self.get_filtered_map(layer='o')[0]
        draw_layer(ax, obstacles, 'black')

        # Agents fields
        for agent in range(self._agent_count):
            start_pos = self.get_start_positions(agent=agent)
            a_c_n_pos = self._map[self._layer_filter(agent=agent)]

            hue = 1.0 / self._agent_count * agent
            color = colorsys.hsv_to_rgb(hue, 1.0, 0.7)
            color_next = colorsys.hsv_to_rgb(hue, 0.1, 0.9)

            # Plot next position
            if self._next_step:
                draw_layer(ax, a_c_n_pos[2], color_next)

            # Plot current position
            draw_layer(ax, a_c_n_pos[1], color)

            # Plot start position
            if start_pos is not None:
                x = start_pos[1] + 0.2
                y = self._size_y - start_pos[0] - 1 + 0.15
                draw_label(ax, x, y, "S", color)

            # Plot aim position
            for y, x in zip(*np.where(a_c_n_pos[0])):
                x = x + 0.2
                y = self._size_y - y - 1 + 0.15
                draw_label(ax, x, y, "E", color)

        # Draw Border
        border = patches.Rectangle((0, 0), self._size_y, self._size_x, linewidth=5, edgecolor='black',
                                   facecolor='none')
        ax.add_patch(border)

        plt.ylim(0, self._size_x)
        plt.xlim(0, self._size_y)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    o = np.array([[0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])
    game_map = Map(6, 6, 2, o, next_step=True)

    game_map.set_aim_positions([[0, 3],
                                [1, 3]])
    game_map.set_current_positions([[0, 0],
                                    [1, 0]])
    game_map.set_current_positions([[0, 1],
                                    [1, 1]])
    game_map.set_next_positions([[0, 2],
                                 [1, 2]])

    print(game_map.get_map())
    # print(game_map.get_map_for_agent(0))
    # print(game_map.get_filtered_map(agent='2', layer='a'))
    game_map.plot_overview()
