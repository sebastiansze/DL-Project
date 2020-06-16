import numpy as np

from map import Map

if __name__ == '__main__':

    # Iterate over games:
    # while True:
    for i in range(5):
        # Define number of agents and map size randomly
        agent_count = np.random.randint(1, 5)
        map_size = np.random.randint(2 + agent_count, 50 + agent_count, 2)

        # Create obstacle map randomly
        obstacle_map_size = np.random.randint((1, 1), map_size - 2, 2)
        obstacle_map = np.random.choice([True, False, False], obstacle_map_size)
        padding_l_t = np.rint((map_size - obstacle_map.shape) / 2).astype(int)
        padding_r_b = np.array(map_size - obstacle_map.shape - padding_l_t, dtype=int)
        obstacle_map = np.pad(obstacle_map, [(padding_l_t[0], padding_r_b[0]),
                                             (padding_l_t[1], padding_r_b[1])])
        print(padding_l_t)
        print(padding_r_b)
        print(map_size)
        print(obstacle_map.shape)
        arena = Map(size_x=map_size[0], size_y=map_size[1], agent_count=agent_count,
                    obstacle_map=obstacle_map, next_step=False)

        # Set start and end positions
        # arena.set_aim_positions(...)
        # arena.set_current_positions(...)

        arena.plot_all(block=True)

        # TODO: To be continued
