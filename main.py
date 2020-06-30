import os
import numpy as np
from datetime import datetime

from network import Network
from map import Map


def get_random_obstacle_map(map_size):
    obstacle_map_size = np.random.randint((1, 1), map_size - 2, 2)
    obstacle_map = np.random.choice([True, False, False], obstacle_map_size)
    padding_l_t = np.rint((map_size - obstacle_map.shape) / 2).astype(int)
    padding_r_b = np.array(map_size - obstacle_map.shape - padding_l_t, dtype=int)
    obstacle_map = np.pad(obstacle_map, [(padding_l_t[0], padding_r_b[0]),
                                         (padding_l_t[1], padding_r_b[1])])
    return obstacle_map


def get_random_start_and_end_position(map_size, agent_count):
    all_border_cells = np.concatenate([[[x, 0] for x in range(0, map_size[0], 2)],
                                       [[x, map_size[1] - 1] for x in range(0, map_size[0], 2)],
                                       [[0, y] for y in range(1, map_size[1] - 1, 2)],
                                       [[map_size[0] - 1, y] for y in range(1, map_size[1] - 1, 2)]])
    np.random.shuffle(all_border_cells)

    return all_border_cells[0:agent_count], all_border_cells[agent_count:agent_count * 2]


def get_truth_reward(next_states_min_distances):
    obstacle_min_dist = next_states_min_distances[:, :, 0]
    other_agent_min_dist = next_states_min_distances[:, :, 1]
    aim_dist = next_states_min_distances[:, :, 2]

    reward = np.where(other_agent_min_dist < 1, -500,
                      np.where(obstacle_min_dist < 1, -250,
                               np.where(aim_dist < 1, 10,
                                        -aim_dist)))
    return reward


def run(agent_count_min=1, agent_count_max=1, map_size_min=25, map_size_max=25, use_obstacles=False):
    # Init network
    net = Network()

    # Iterate over games:
    i_game = -1

    while True:
        i_game += 1
        game_dt = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        print('Game Nr. {}'.format(i_game))
        print('started at {}'.format(game_dt))

        # Define number of agents randomly in given range
        if agent_count_min == agent_count_max:
            agent_count = agent_count_min
        else:
            agent_count = np.random.randint(agent_count_min, agent_count_max)
        print('Number of Agents: {}'.format(agent_count))

        # Define size of map randomly in given range
        # If min or max limit parameter is None select a useful limit automatically
        map_size_min = map_size_min if map_size_min is not None else 3 + agent_count
        map_size_max = map_size_max if map_size_max is not None else 20 + agent_count
        if map_size_min == map_size_max:
            map_size = np.array([map_size_min, map_size_min])
        else:
            map_size = np.random.randint(map_size_min, map_size_max, 2)
        print('Size of Arena: {}'.format(map_size))

        # Create obstacle map randomly if wanted
        obstacle_map = get_random_obstacle_map(map_size) if use_obstacles else None

        # Initialize a new game arena
        arena = Map(size_x=map_size[0], size_y=map_size[1], agent_count=agent_count,
                    obstacle_map=obstacle_map, next_step=False)

        # Set start and end positions randomly
        start_positions, end_positions = get_random_start_and_end_position(map_size, agent_count)
        arena.set_aim_positions(start_positions)
        arena.set_current_positions(end_positions)

        # Generate Agents
        agents = net.generate_agents(agent_count)

        # Check if directory for images exists
        img_game_dir = os.path.join('img', '{}_game_{}'.format(game_dt, i_game))
        if not os.path.exists(img_game_dir):
            os.makedirs(img_game_dir)

        # Save image for t = 0
        # arena.plot_overview(save_as=os.path.join(img_game_dir, 'time_{}.png'.format(0)))

        # Start playing
        time_step = 0
        while arena.is_anyone_still_moving():
            time_step += 1
            print('  - time step: {}'.format(time_step))

            # Apply network for each agent independently
            commands = []
            for i_agent, agent in enumerate(agents):
                agent_map = arena.get_map_for_agent(i_agent)
                command = agent.predict(agent_map)
                commands.append(command)

            # Apply network output
            arena.move_agents(commands)

            # Print Agent status
            print('    - Agent Status: {}'.format(arena.get_agents_conditions()))
            print('    - Agent Duration: {}'.format(arena.get_agents_durations_since_start()))
            print('    - Agent Distance: {}'.format(arena.get_agents_distances_since_start()))

            # Save image
            arena.plot_overview(save_as=os.path.join(img_game_dir, 'time_{}.png'.format(time_step)))

            # Reward and penalty
            truth = get_truth_reward(arena.get_agents_next_states_min_distances())
            # ...

        print('Time Steps: {}'.format(time_step))
        print()


if __name__ == '__main__':
    ##############################
    #   decide and comment the   #
    # other version accordingly: #
    ##############################

    # Run the simplest version:
    run()

    # Run a more complex version:
    # rand_agent_count_min = 1
    # rand_agent_count_max = 5
    # rand_map_size_min = None  # Choose automatically
    # rand_map_size_max = None  # Choose automatically
    # run_with_obstacles = True
    # run(rand_agent_count_min, rand_agent_count_max, rand_map_size_min, rand_map_size_max, run_with_obstacles)
