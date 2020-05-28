import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Map:
    def __init__(self, size_x, size_y, nr_player=1, obstacle_map=None, next_step=False):
        self._size_x = size_x
        self._size_y = size_y
        self._nr_player = nr_player
        self._next_step = next_step

        # Check input values
        if nr_player < 1:
            raise ValueError('number of agents must be at least one')
        if size_x * size_y < 2 * nr_player:
            raise ValueError('board size is to small')

        # Create layer array of type char to filter the map
        self._layers = np.array([0, 'o'], dtype=np.dtype('U1'))  # obstacles
        for p in range(nr_player):
            unstacked_layers = [self._layers,
                                np.array([p + 1, 'a'], dtype=np.dtype('U1')),  # aim position
                                np.array([p + 1, 'c'], dtype=np.dtype('U1'))]  # current position
            if next_step:
                unstacked_layers.append(np.array([p + 1, 'n'], dtype=np.dtype('U1')))  # next position
            self._layers = np.vstack(unstacked_layers)

        # Create map skeleton
        nr_layers = 1 + nr_player * (3 if next_step else 2)
        self._map = np.zeros((nr_layers, size_x, size_y), dtype=bool)

        # Add obstacles to map
        if obstacle_map is not None:
            if not isinstance(obstacle_map, np.ndarray):
                raise TypeError('obstacle map must ba a numpy array')
            if obstacle_map.shape == (size_x, size_y):
                self._map[0, :, :] = obstacle_map
            else:
                raise ValueError('obstacle map must have the same size as main map')

        # TODO: History
        self._hist = None

    def _layer_filter(self, agent=None, layer=None):
        agent = None if agent is None else str(int(agent) + 1)  # in self._layers agent number starts at one
        agent_filter = [True for _ in range(self._layers.shape[0])] if agent is None else self._layers[:, 0] == agent
        layer_filter = [True for _ in range(self._layers.shape[0])] if layer is None else self._layers[:, 1] == layer
        return np.array(agent_filter) & np.array(layer_filter)

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
        :param positions: (boolean) numpy array with shape [nr_agents * 2] and x and y coordinates inside for all agents
        """
        if layer == 'n' and not self._next_step:
            raise Exception('Next position not activated in this map')

        positions = np.concatenate([np.expand_dims(np.arange(self._nr_player), 1), positions], 1)
        ary = np.zeros((self._nr_player, self._size_x, self._size_y))
        ary[positions[:, 0], positions[:, 1], positions[:, 2]] = [True for _ in range(self._nr_player)]
        self._map[self._layer_filter(layer=layer)] = ary

    def set_aim_positions(self, positions):
        """
        Define aim positions for all agents
        :param positions: (boolean) numpy array with shape [nr_agents * 2] and x and y coordinates inside for all agents
        """
        self.set_positions(layer='a', positions=positions)

    def set_current_positions(self, positions):
        """
        Define current positions for all agents
        :param positions: (boolean) numpy array with shape [nr_agents * 2] and x and y coordinates inside for all agents
        """
        self.set_positions(layer='c', positions=positions)

    def set_next_positions(self, positions):
        """
        Define next positions for all agents
        :param positions: (boolean) numpy array with shape [nr_agents * 2] and x and y coordinates inside for all agents
        """
        self.set_positions(layer='n', positions=positions)

    def get_map(self):
        return copy.deepcopy(self._map)

    def get_filtered_map(self, agent=None, layer=None):
        return copy.deepcopy(self._map[self._layer_filter(agent, layer)])

    def get_map_for_agent(self, agent, view_filed):
        obstacles = self._map[self._layer_filter(layer='o')]
        a_c_n_pos = self._map[self._layer_filter(agent=agent)]
        other_pos = self._map[self._layer_filter(layer='c')]
        other_pos = other_pos & ~self._map[self._layer_filter(agent=agent, layer='c')]  # subtract own current position
        # TODO: field of view
        # other_pos = other_pos & ...
        return np.concatenate((obstacles, a_c_n_pos, other_pos))

    def plot_overview(self):
        def draw_layer(ax, layer, color):
            for x in range(self._size_x):
                for y in range(self._size_y):
                    if layer[x, y]:
                        rect = patches.Rectangle((y, self._size_x - x - 1), 1, 1, linewidth=0, edgecolor='none',
                                                 facecolor=color)
                        ax.add_patch(rect)

        # Create figure and axes
        fig, ax = plt.subplots(1)

        # Draw Border
        border = patches.Rectangle((0, 0), self._size_y, self._size_x, linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(border)

        # Obstacles
        obstacles = self.get_filtered_map(layer='o')[0]
        print(obstacles)
        draw_layer(ax, obstacles, 'black')

        plt.ylim(0, self._size_x)
        plt.xlim(0, self._size_y)
        plt.show()


if __name__ == '__main__':
    o = np.array([[0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])
    game_map = Map(6, 6, 2, o)

    # game_map.set_position(agent=0, layer='a', x=5, y=5)
    # game_map.set_position(agent=1, layer='a', x=3, y=5)
    # game_map.set_position(agent=0, layer='c', x=0, y=0)
    # game_map.set_position(agent=1, layer='c', x=0, y=5)

    game_map.set_aim_positions([[5, 5],
                                [3, 5]])

    game_map.set_current_positions([[0, 0],
                                    [0, 5]])

    print(game_map.get_map())
    # print(game_map.get_filtered_map(agent='2', layer='a'))
    # game_map.plot_overview()
    # game_map.get_map_for_agent(0)
