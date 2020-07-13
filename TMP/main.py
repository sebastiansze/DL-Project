import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from Agent import Agent
from map import Map, print_layers

# TODO
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


#################################
#   Define parameters manually  #
#################################
def define_parameters():
    params = dict()
    params['map_size_max'] = 15
    params['with_stay_action'] = False
    params['time_step_max'] = 4 * params['map_size_max']
    params['episodes'] = 100

    params['epsilon'] = 1.0
    params['eps_min'] = 0.01
    params['eps_dec'] = 1e-4
    params['gamma'] = 0.99
    params['learning_rate'] = 5e-4
    params['first_layer_size'] = 128  # neurons in the first layer
    params['second_layer_size'] = 256  # neurons in the second layer
    params['third_layer_size'] = 128  # neurons in the third layer
    params['memory_size'] = 100000
    params['batch_size'] = 64
    params['replace'] = 100

    params['weights_path'] = 'weights'
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


def run(params):
    agents = Agent(params)

    weights_filepath = params['weights_path']
    if params['load_weights']:
        agents.load_models()  # TODO: Path (weights_filepath)
        print("weights loaded")

    conditions = []
    errors = []
    all_action_list = []
    scores, eps_hist = [], []

    score_saver = []
    avg_score_saver = []
    ddqn_scores = []
    eps_history = []
    savedGames = []
    prec = 20
    reached = 0
    reached_last_100 = 0

    # Iterate over games:
    i_game = -1
    while i_game < params['episodes'] - 1:
        i_game += 1
        score = 0
        game_dt = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        print('Game Nr. {}'.format(i_game))
        # print('started at {}'.format(game_dt))

        # Define number of agents randomly in given range
        if params['agent_count_min'] == params['agent_count_max']:
            agent_count = params['agent_count_min']
        else:
            agent_count = np.random.randint(params['agent_count_min'], params['agent_count_max'])
        # print('Number of Agents: {}'.format(agent_count))

        # Define size of map randomly in given range
        # If min or max limit parameter is None select a useful limit automatically
        map_size_min = params['map_size_min'] if params['map_size_min'] is not None else 3 + agent_count
        map_size_max = params['map_size_max'] if params['map_size_max'] is not None else 20 + agent_count
        if map_size_min == map_size_max:
            map_size = np.array([map_size_min, map_size_min])
        else:
            map_size = np.random.randint(map_size_min, map_size_max, 2)
        # print('Size of Arena: {}'.format(map_size))

        # Create obstacle map randomly if wanted
        obstacle_map = get_random_obstacle_map(map_size) if params['use_obstacles'] else None

        # Initialize a new game arena
        arena = Map(size_x=map_size[0], size_y=map_size[1], time_step_max=params['time_step_max'],
                    agent_count=agent_count, obstacle_map=obstacle_map, next_step=False)

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
        game_sav = []
        avg_score = 0
        losses_sum = 0
        while arena.is_anyone_still_moving():
            time_step += 1
            time_steps = np.tile(time_step, (agent_count,1))
            # print('  - time step: {}'.format(time_step))

            # get old (current) observations
            old_observations = arena.get_map_for_all_agent()

            # perform new actions and get the new observations and rewards
            actions = agents.choose_actions(old_observations)
            if np.any(np.sum(actions, axis=1) != 1):
                raise ValueError('one hot vector invalid.')
            arena.move_agents(actions)
            new_observations = arena.get_map_for_all_agent()
            rewards = arena.get_rewards()

            # store game statistics
            reached += np.count_nonzero(arena.get_agents_conditions() == 'a')
            if i_game > (params['episodes'] - 100):
                reached_last_100 += np.count_nonzero(arena.get_agents_conditions() == 'a')
            score += np.sum(rewards) / agent_count

            # perform agents training
            agents.store_transition(old_observations, actions, rewards, new_observations, arena.which_agent_is_done())
            agents.learn()

            game_sav.append(old_observations)
            eps_history.append(agents.epsilon)

            ddqn_scores.append(score)
            if i_game > 20:
                avg_score = np.mean(ddqn_scores[-10])

            # Print Agent status
            # print('    - Agent Status: {}'.format(arena.get_agents_conditions()))
            # print('    - Agent Duration: {}'.format(arena.get_agents_durations_since_start()))
            # print('    - Agent Distance: {}'.format(arena.get_agents_distances_since_start()))

            # Save image
            # arena.plot_overview(save_as=os.path.join(img_game_dir, 'time_{}.png'.format(time_step)))

        score_saver.append(score)
        if i_game > 20:
            avg_score_saver.append(avg_score)

        savedGames.append(game_sav)
        if i_game % int(params['episodes'] / prec) == int(params['episodes'] / prec) - 1:
            print('episode: ', i_game, 'score: %.2f' % score,
                  ' average score %.2f' % avg_score,
                  'Epsilon %.3f' % agents.epsilon)

        conditions.append(arena.get_agents_conditions())
        print('Time Steps: {}'.format(time_step))
        print('agent status: {}'.format(arena.get_agents_conditions()))
        if 'a' in arena.get_agents_conditions():
            print('  => AIM ACHIEVED!')
        # print('Error: {}'.format(error))
        print()

        if i_game % 10 == 9:
            arena.plot_overview(save_as=os.path.join('img', '{}_game_{}_time_{}.png'.format(game_dt,
                                                                                            i_game,
                                                                                            time_step)))
            if params['train']:
                agents.save_models()  # TODO: Path (params['weights_path'])

    conditions = np.reshape(conditions, -1)
    cond_unique, cond_count = np.unique(conditions, return_counts=True)
    cond_prop = cond_count * 100 / conditions.shape[0]

    all_action_list = np.reshape(all_action_list, (-1, 5)).astype('bool').T
    all_action_list = np.select(all_action_list, ['STAY', 'UP', 'RIGHT', 'DOWN', 'LEFT'])
    act_unique, act_count = np.unique(all_action_list, return_counts=True)
    act_prop = act_count * 100 / all_action_list.shape[0]

    print('=======================')
    print('Conditions: {}'.format(', '.join(['{}: {}%'.format(u, c) for u, c in zip(cond_unique, cond_prop)])))
    print('Used actions: {}'.format(', '.join(['{}: {}%'.format(u, c) for u, c in zip(act_unique, act_prop)])))

    print()
    print(str(params['episodes']) + " Spieldurchläufe: " + str(reached) + " mal Ziel erreicht, Quote = " + str(reached / params['episodes']))
    print("Quote der letzten 100 Durchläufe " + str(reached_last_100 / 100))

    plt.plot(errors)


if __name__ == '__main__':
    ##############################
    #   decide and comment the   #
    # other version accordingly: #
    ##############################

    # Run the simplest version:
    parameters = define_parameters()
    parameters['map_size_min'] = 15
    parameters['agent_count_min'] = 1
    parameters['agent_count_max'] = 1
    parameters['use_obstacles'] = False
    run(parameters)

    # Run a more complex version:
    # rand_agent_count_min = 1
    # rand_agent_count_max = 5
    # rand_map_size_min = None  # Choose automatically
    # rand_map_size_max = None  # Choose automatically
    # run_with_obstacles = True
    # run(rand_agent_count_min, rand_agent_count_max, rand_map_size_min, rand_map_size_max, run_with_obstacles)
