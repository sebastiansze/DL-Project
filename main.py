import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from Agent import Agent
from GameLogic import Game, Point
from visualisation import Visualisation

MAX_REWARD = 200000


def play():
    pass


def train(n_games=2500, env_size=(15, 15), n_agents=2, timeout=100, resume=False, view_reduced=True, view_size=(2, 2, 2, 2)):
    print(f"------------------------------------------------------------------------------------------------")
    print(f"Starting training for {n_games} with {n_agents} agents...")
    print(f"Settings:")
    print(f"World size:\t{env_size}\nTimeout:\t{timeout}\nReduced view:\t{view_reduced}\nView size:\t{view_size}")
    print(f"------------------------------------------------------------------------------------------------")

    score_saver = []
    avg_score_saver = []
    ddqn_scores = []
    eps_history = []
    saved_games = []
    saved_obstacles = []
    prec = 40
    reached = np.zeros(n_agents, dtype=np.int32)
    reached_last_100 = np.zeros(n_agents, dtype=np.int32)

    if view_reduced:
        input_size = (view_size[0] + 1 + view_size[1]) * (view_size[2] + 1 + view_size[3]) + 4
    else:
        input_size = env_size[0] * env_size[1]
    agents = []

    for agent_id in range(n_agents):
        agent = Agent(f"agent_{agent_id}", gamma=0.99, epsilon=1.0, lr=1 * 5e-3, n_actions=4,
                      input_dims=[input_size], mem_size=100000, batch_size=64,
                      eps_min=0.01, eps_dec=5 * 1e-5, replace=100)
        if resume:
            agent.load_models()
        agents.append(agent)

    # Main training loop
    for i_game in tqdm(range(n_games)):
        scores = np.zeros(n_agents)
        avg_scores = np.zeros(n_agents)
        agent_in_final_state = np.full(n_agents, False)

        num_obstacles = np.random.randint(15, 25) - 2 * n_agents
        obstacles = []
        for i in range(num_obstacles):
            obstacles.append(Point(np.random.randint(1, env_size[0]), np.random.randint(1, env_size[1])))
        saved_obstacles.append(np.array([o.to_numpy() for o in obstacles]))

        env = Game(obstacles, None, env_size, MAX_REWARD, view_reduced=view_reduced, view_size=view_size)
        for i in range(n_agents):
            env.add_player()

        observations = env.reset()
        game_sav = [observations]
        time_step = 0
        # Run until all agents reached a final state
        while not np.all(agent_in_final_state):
            time_step += 1
            # Obtain actions for each agent
            actions = []
            # Get actions from all agents that are not in a final state
            for agent_id, agent in enumerate(agents):
                if not agent_in_final_state[agent_id]:
                    actions.append(agent.choose_action(observations[agent_id]))
                else:
                    actions.append(None)
            # Execute actions on board
            next_observations, rewards, agent_in_final_state = env.step(actions)
            # Save history for each agent and optimize
            for agent, observation, action, reward, next_observation, is_in_final_state in \
                    zip(agents, observations, actions, rewards, next_observations, agent_in_final_state):
                # Only store and optimize if the agent did something
                if action is not None:
                    agent.store_transition(observation, action, reward, next_observation, int(is_in_final_state))
                    agent.learn()

            for agent_id, action in enumerate(actions):
                if action is not None and rewards[agent_id] == MAX_REWARD:
                    reached[agent_id] += 1
                    if i_game > (n_games - 100):
                        reached_last_100[agent_id] += 1

            scores += rewards
            observations = next_observations
            game_sav.append(next_observations)
            eps_history.append([agent.epsilon for agent in agents])
            ddqn_scores.append(scores)

            if time_step == timeout:
                agent_in_final_state = np.full(n_agents, True)

            if i_game > 0 and i_game % 10 == 0:
                for agent in agents:
                    agent.save_models()

            if all(agent_in_final_state) and i_game > 20:
                avg_scores = np.mean(ddqn_scores[:-10], axis=0)
        score_saver.append(scores)
        if i_game > 20:
            avg_score_saver.append(avg_scores)
            epsilons = {agent.id: agent.epsilon for agent in agents}
            if i_game % int(n_games / prec) == int(n_games / prec) - 1:
                print(f"episode: {i_game} score: {np.round(scores.tolist(),3)}, average score {avg_scores.tolist()} "
                      f"epsilon {epsilons} Erreicht: {reached.tolist()}")
        saved_games.append(game_sav)

    print(f"\n{n_games} Spieldurchläufe: {reached.tolist()} mal Ziel erreicht - Qoute: {(reached / n_games).tolist()}")
    print("Quote der letzten 100 Durchläufe " + str((reached_last_100 / 100).tolist()))
    # plt.plot(score_saver)
    # plt.show()
    # plt.plot(avg_score_saver)
    # plt.show()

    plot_game_i_list = np.arange(n_games - 1, 0, - n_games * 0.1)
    plot_game_i_list = np.concatenate([plot_game_i_list, np.argsort(-1 * np.max(score_saver, axis=1))[:5]])
    plot_game_i_list = np.unique(plot_game_i_list)
    print('Visualize this games:{}'.format(plot_game_i_list))
    dt = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    for i_game in plot_game_i_list:
        viz = Visualisation(saved_games[i_game], env_size, n_agents,
                            view_padding=view_size, view_reduced=view_reduced,
                            truth_obstacles=saved_obstacles[i_game])
        # if viz.time_steps > 30:
        print(f'Generate visual output for game {i_game} of session {dt}...')
        # viz.plot_all(plot_input=True, save_as=os.path.join('img', f'{dt}_game_{i_game}.png'))
        viz.save_all_as_video(dt, i_game, plot_input=True)
        # viz.save(dt, i_game)

    # What was this supposed to do? Definitely does not work like this!
    # helper = Helpers(env)
    # randomgame = np.random.randint(1,20)
    # print(randomgame)
    #
    # fig, ax = plt.subplots(figsize=(env_size[1]+3, env_size[0]))
    # sns.heatmap(helper.show_reward(saved_games[-randomgame][0],elemMax=True).reshape(boardSize[0], boardSize[1]), annot=True, fmt="d")
    # plt.show()

    print('Done')


if __name__ == '__main__':
    train()
