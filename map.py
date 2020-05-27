import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Map:
    def __init__(self, size_x, size_y, nr_player=1, obstacle_map=None, next_step=False):
        self._size_x = size_x
        self._size_y = size_y
        self.nr_player = nr_player

        # Check input values
        if nr_player < 1:
            raise ValueError('number of agents must be at least one')
        if size_x * size_y < 2 * nr_player:
            raise ValueError('board size is to small')

        # Create layer array to filter the map
        self._layers = np.array([0, 'o'], dtype=np.dtype('U1'))                        # obstacles
        for p in range(nr_player):
            unstacked_layers = [self._layers,
                                np.array([p + 1, 'a'], dtype=np.dtype('U1')),          # aim position
                                np.array([p + 1, 'c'], dtype=np.dtype('U1'))]          # current position
            if next_step:
                unstacked_layers.append(np.array([p + 1, 'n'], dtype=np.dtype('U1')))  # next position
            self._layers = np.vstack(unstacked_layers)

        # Create map skeleton
        nr_layers = 1 + nr_player * (3 if next_step else 2)
        self._map = np.zeros((nr_layers, size_x, size_y))

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

    def set_aim_positions(self, positions):
        # TODO
        pass

    def set_current_positions(self, positions):
        for agent, position in enumerate(positions):
            # TODO
            pass

    def set_next_positions(self, positions):
        # TODO
        pass

    def get_map(self):
        return self._map

    def get_filtered_map(self, agent=None, layer=None):
        agent_filter = [True for _ in range(self._layers.shape[0])] if agent is None else self._layers[:, 0] == agent
        layer_filter = [True for _ in range(self._layers.shape[0])] if layer is None else self._layers[:, 1] == layer
        combi_filter = np.array(agent_filter) & np.array(layer_filter)
        return self._map[combi_filter]

    def get_map_for_agent(self, agent=0):
        # TODO
        pass

    def plot_overview(self):
        def draw_layer(ax, layer, color):
            for x in range(self._size_x):
                for y in range(self._size_y):
                    if layer[x, y]:
                        rect = patches.Rectangle((y, self._size_x-x-1), 1, 1, linewidth=0, edgecolor='none', facecolor=color)
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
    # o = np.random.randint(low=0, high=2, size=(5, 5))
    o = np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]])
    print(o)
    game_map = Map(6, 5, 2, o)
    print(game_map.get_map())
    # print(game_map.get_filtered_map(agent='2', layer='a'))
    game_map.plot_overview()
