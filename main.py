import os
import numpy as np
from datetime import datetime

from network import Network
from map import Map

if __name__ == '__main__':

    # Init network
    net = Network()

    # Iterate over games:
    i_game = -1
    # for i_game in range(15):
    while True:
        i_game += 1
        game_dt = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        print('Game Nr. {}'.format(i_game))
        print('started at {}'.format(game_dt))

        # Define number of agents and map size randomly
        agent_count = 1
        map_size = np.array([25, 25])

        # Create obstacle map randomly
        arena = Map(size_x=map_size[0], size_y=map_size[1], agent_count=agent_count, next_step=False)

        # Set start and end positions randomly
        all_border_cells = np.concatenate([[[x, 0] for x in range(0, map_size[0], 2)],
                                           [[x, map_size[1] - 1] for x in range(0, map_size[0], 2)],
                                           [[0, y] for y in range(1, map_size[1] - 1, 2)],
                                           [[map_size[0] - 1, y] for y in range(1, map_size[1] - 1, 2)]])
        np.random.shuffle(all_border_cells)
        arena.set_aim_positions(all_border_cells[0:agent_count])
        arena.set_current_positions(all_border_cells[agent_count:agent_count * 2])

        # Generate Agents
        agents = net.generate_agents(agent_count)

        # Check if directory for images exists
        # img_game_dir = os.path.join('img', '{}_game_{}'.format(game_dt, i_game))
        # if not os.path.exists(img_game_dir):
        #     os.makedirs(img_game_dir)

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
                command = agent.move(agent_map)
                commands.append(command)

            # Apply network output
            arena.move_agents(commands)

            # Print Agent status
            print('    - Agent Status: {}'.format(arena.get_agents_status()))
            print('    - Agent Duration: {}'.format(arena.get_agents_duration()))
            print('    - Agent Distance: {}'.format(arena.get_agents_distance()))

            # Save image
            # arena.plot_overview(save_as=os.path.join(img_game_dir, 'time_{}.png'.format(time_step)))

        # Reward and penalty
        net.backward(agent_status=arena.get_agents_status(),
                     durations=arena.get_agents_duration(),
                     distances=arena.get_agents_distance())

        print('Time Steps: {}'.format(time_step))
        print()
