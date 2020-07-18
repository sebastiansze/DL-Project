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
VIEW_RANGE = (2, 2, 2, 2)
VIEW_REDUCED = True


def play():
    pass


def train(n_games=200, env_size=(15, 15), n_agents=5, timeout=100, resume=False):
    score_saver = []
    avg_score_saver = []
    ddqn_scores = []
    eps_history = []
    saved_games = []
    prec = 40
    reached = np.zeros(n_agents, dtype=np.int32)
    reached_last_100 = np.zeros(n_agents, dtype=np.int32)

    if VIEW_REDUCED:
        input_size = (VIEW_RANGE[0] + 1 + VIEW_RANGE[1])*(VIEW_RANGE[2] + 1 + VIEW_RANGE[3]) + 4
        print("Use Reduced View Mode")
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

    num_obstacles = np.random.randint(15, 25)
    obstacles = []
    for i in range(num_obstacles):
        obstacles.append(Point(np.random.randint(1, env_size[0]), np.random.randint(1, env_size[1])))

    # Main training loop
    for i_game in tqdm(range(n_games)):
        scores = np.zeros(n_agents)
        avg_scores = np.zeros(n_agents)
        agent_in_final_state = np.full(n_agents, False)
        env = Game(obstacles, None, env_size, MAX_REWARD, viewReduced=VIEW_REDUCED, viewSize=VIEW_RANGE)
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

    plot_game_i_list = range(0, n_games, 20) # np.argsort(-1 * np.max(score_saver, axis=1))[:5]
    dt = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    for i_game in plot_game_i_list:
        path = os.path.join('img', f'{dt}_game_{i_game}.mp4')
        print(f'Generate video {path}...')
        viz = Visualisation(saved_games[i_game], env_size, n_agents, view_padding=VIEW_RANGE, view_reduced=VIEW_REDUCED)
        viz.save_all_as_video(plot_input=True, save_as=path)

    # What was this supposed to do? Definitely does not work like this!
    # helper = Helpers(env)
    # randomgame = np.random.randint(1,20)
    # print(randomgame)
    #
    # fig, ax = plt.subplots(figsize=(env_size[1]+3, env_size[0]))
    # sns.heatmap(helper.show_reward(saved_games[-randomgame][0],elemMax=True).reshape(boardSize[0], boardSize[1]), annot=True, fmt="d")
    # plt.show()


if __name__ == '__main__':
    train()
