#!/usr/bin/env python3
import os
import numpy as np
import click
import torch
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from itertools import count

from network import Network, ReplayMemory, Transition
from map import Map, print_layers


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


def get_reward(conditions, distances_to_aims, max_distance):
    condition_list = [conditions == c for c in ['a', 's', '3', 't']]
    choice_list = [+2 * max_distance,  # aim
                   -2 * max_distance,  # self inflicted accident
                   -1 * max_distance,  # third-party fault accident
                   -1 * max_distance]  # time out
    # dist_func = 1 / (2 * distances_to_aims + 1) + 0.5
    # return np.select(condition_list, [1, 0, 0.2], dist_func)
    dist_func = max_distance - distances_to_aims
    return np.select(condition_list, choice_list, dist_func)


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
            truth = get_reward(arena.get_agents_conditions(), arena.get_agents_next_states_min_distances())
            # ...

        print('Time Steps: {}'.format(time_step))
        print()


steps_done = 0


@click.command()
def train(episodes=550, batch_size=128, gamma=0.999, eps_start=0.9, eps_end=0.05, eps_decay=200, target_update=10):
    map_size = np.array([15, 15])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = Network().to(device)
    target_net = Network().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()



    memory = ReplayMemory(3000)
    optimizer = optim.RMSprop(policy_net.parameters())

    def select_action(state):
        global steps_done
        eps_treshold = eps_end + (eps_start - eps_end) * np.exp(-1. * steps_done / eps_decay)
        steps_done += 1
        if np.random.random() > eps_treshold:
            with torch.no_grad():
                action = policy_net(torch.from_numpy(state).float().to(device))
                one_hot = torch.zeros(5)
                one_hot[torch.argmax(action)] = 1
                return np.asarray(one_hot.int())
        else:
            ret = np.array([1, 0, 0, 0, 0])
            np.random.shuffle(ret)
            return ret

    def optimize_model():
        if len(memory) < batch_size:
            return
        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([torch.from_numpy(s).float() for s in batch.next_state
                                           if s is not None]).to(device)

        state_batch = torch.from_numpy(np.concatenate(batch.state)).float().to(device)
        action_batch = torch.from_numpy(np.concatenate(batch.action)).long().to(device)
        reward_batch = torch.from_numpy(np.concatenate(batch.reward)).float().to(device)

        state_action_values = policy_net(state_batch).gather(1, action_batch.argmax(dim=1).reshape((batch_size, 1)))

        next_state_values = torch.zeros(batch_size, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    for i in range(episodes):
        env = Map(size_x=map_size[0], size_y=map_size[1], agent_count=1,
                  obstacle_map=get_random_obstacle_map(map_size), next_step=False)
        env.set_aim_positions(np.array([[np.random.randint(0, map_size[0]), np.random.randint(0, map_size[1])]]))
        env.set_current_positions(np.array([[np.random.randint(0, map_size[0]), np.random.randint(0, map_size[1])]]))
        state = env.get_map().reshape(1, 3, map_size[0], map_size[1])
        for t in count():
            action = select_action(state).reshape(1, 5)
            env.move_agents(action)
            agent_conditions = env.get_agents_conditions()
            done = np.all(agent_conditions != "")
            if not done:
                next_state = env.get_map().reshape(1, 3, map_size[0], map_size[1])
            else:
                next_state = None
            reward = get_reward(agent_conditions, env.get_distances_to_aims(), np.sqrt(np.sum(np.square(map_size))))
            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model()
            if done:
                if i > 500:
                    env.plot_all()
                break
        if i % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())





if __name__ == '__main__':
    ##############################
    #   decide and comment the   #
    # other version accordingly: #
    ##############################

    # Run the simplest version:
    #run()
    train()

    # Run a more complex version:
    # rand_agent_count_min = 1
    # rand_agent_count_max = 5
    # rand_map_size_min = None  # Choose automatically
    # rand_map_size_max = None  # Choose automatically
    # run_with_obstacles = True
    # run(rand_agent_count_min, rand_agent_count_max, rand_map_size_min, rand_map_size_max, run_with_obstacles)
