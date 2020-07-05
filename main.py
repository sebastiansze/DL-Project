import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from Agents import Agents
from map import Map, print_layers


#################################
#   Define parameters manually  #
#################################
def define_parameters():
    params = dict()
    params['epsilon_decay_linear'] = 1 / 75
    params['learning_rate'] = 0.0005
    params['first_layer_size'] = 150  # neurons in the first layer
    params['second_layer_size'] = 150  # neurons in the second layer
    params['third_layer_size'] = 150  # neurons in the third layer
    params['episodes'] = 150
    params['memory_size'] = 2500
    params['batch_size'] = 500
    params['weights_path'] = 'weights/weights.hdf5'
    params['load_weights'] = True
    params['train'] = True
    return params


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


def get_one_hot_vectors(indices, length=5):
    return np.array([np.eye(1, length, int(x))[0] for x in indices])


# def initialize_game(map, batch_size):
#     state_init1 = agent.get_state(game, player, food)  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
#     action = [1, 0, 0]
#     player.do_move(action, player.x, player.y, game, food, agent)
#     state_init2 = agent.get_state(game, player, food)
#     reward1 = agent.set_reward(player, game.crash)
#     agent.remember(state_init1, action, reward1, state_init2, game.crash)
#     agent.replay_new(agent.memory, batch_size)


def run(params):
    agents = Agents()
    errors = []

    # Iterate over games:
    # i_game = -1

    for i_game in range(200):
        # i_game += 1
        game_dt = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        print('Game Nr. {}'.format(i_game))
        print('started at {}'.format(game_dt))

        # Define number of agents randomly in given range
        if params['agent_count_min'] == params['agent_count_max']:
            agent_count = params['agent_count_min']
        else:
            agent_count = np.random.randint(params['agent_count_min'], params['agent_count_max'])
        print('Number of Agents: {}'.format(agent_count))

        # Define size of map randomly in given range
        # If min or max limit parameter is None select a useful limit automatically
        map_size_min = params['map_size_min'] if params['map_size_min'] is not None else 3 + agent_count
        map_size_max = params['map_size_max'] if params['map_size_max'] is not None else 20 + agent_count
        if map_size_min == map_size_max:
            map_size = np.array([map_size_min, map_size_min])
        else:
            map_size = np.random.randint(map_size_min, map_size_max, 2)
        print('Size of Arena: {}'.format(map_size))

        # Create obstacle map randomly if wanted
        obstacle_map = get_random_obstacle_map(map_size) if params['use_obstacles'] else None

        # Initialize a new game arena
        arena = Map(size_x=map_size[0], size_y=map_size[1], agent_count=agent_count,
                    obstacle_map=obstacle_map, next_step=False)

        # Set start and end positions randomly
        start_positions, end_positions = get_random_start_and_end_position(map_size, agent_count)
        arena.set_aim_positions(start_positions)
        arena.set_current_positions(end_positions)

        # Check if directory for images exists
        # img_game_dir = os.path.join('img', '{}_game_{}'.format(game_dt, i_game))
        # if not os.path.exists(img_game_dir):
        #     os.makedirs(img_game_dir)

        # Save image for t = 0
        # arena.plot_overview(save_as=os.path.join(img_game_dir, 'time_{}.png'.format(0)))

        # Start playing
        time_step = 0
        losses_sum = 0
        while arena.is_anyone_still_moving():
            time_step += 1
            # print('  - time step: {}'.format(time_step))

            if not params['train']:
                epsilon = 0
            else:
                # agent.epsilon is set to give randomness to actions
                epsilon = 1 - (i_game * params['epsilon_decay_linear'])

            # get old state
            old_states = arena.get_map_for_all_agent()
            # perform random actions based on epsilon, or choose the action
            predictions = agents.predict(old_states)
            actions = np.where(np.random.randint(1, size=(agent_count, 1)) < epsilon,  # Choose randomly between ...
                               get_one_hot_vectors(np.random.randint(5, size=agent_count)),  # random actions and ...
                               get_one_hot_vectors(np.argmax(predictions, axis=1)))  # predicted actions.

            # perform new move and get new states
            arena.move_agents(actions)
            new_states = arena.get_map_for_all_agent()

            # set reward for the new state
            rewards = arena.get_reward()

            if params['train']:
                # train short memory base on the new action and state
                agents.train_short_memory(old_states, actions, rewards, new_states, arena.get_agents_conditions())
                # store the new data into a long term memory
                agents.remember(old_states, actions, rewards, new_states, arena.get_agents_conditions())
            #losses_sum += agents.train(y_pred=pred_rewards, y_true=true_rewards)

            if params['train']:
                agents.replay_new(agents.memory, params['batch_size'])

            # Print Agent status
            # print('    - Agent Status: {}'.format(arena.get_agents_conditions()))
            # print('    - Agent Duration: {}'.format(arena.get_agents_durations_since_start()))
            # print('    - Agent Distance: {}'.format(arena.get_agents_distances_since_start()))

            # Save image
            # arena.plot_overview(save_as=os.path.join(img_game_dir, 'time_{}.png'.format(time_step)))


        error = losses_sum / (time_step + 1)
        errors.append(error)
        print('Time Steps: {}'.format(time_step))
        print('Error: {}'.format(error))
        print()

        if i_game % 10 == 9:
            arena.plot_overview(save_as=os.path.join('img', '{}_game_{}_time_{}.png'.format(game_dt,
                                                                                            i_game,
                                                                                            time_step)))

    plt.plot(errors)


if __name__ == '__main__':
    ##############################
    #   decide and comment the   #
    # other version accordingly: #
    ##############################

    # Run the simplest version:
    parameters = define_parameters()
    parameters['agent_count_min'] = 10
    parameters['agent_count_max'] = 10
    parameters['map_size_min'] = 25
    parameters['map_size_max'] = 25
    parameters['use_obstacles'] = False
    run(parameters)

    # Run a more complex version:
    # rand_agent_count_min = 1
    # rand_agent_count_max = 5
    # rand_map_size_min = None  # Choose automatically
    # rand_map_size_max = None  # Choose automatically
    # run_with_obstacles = True
    # run(rand_agent_count_min, rand_agent_count_max, rand_map_size_min, rand_map_size_max, run_with_obstacles)
