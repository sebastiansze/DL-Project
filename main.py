import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

# import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from Agent import Agent
from GameLogic import Game, Point
from visualisation import Helpers

MAX_REWARD = 20000


def play():
    pass


def train(n_games=1000, env_size=(15, 15), timeout=60, resume=False):
    score_saver = []
    avg_score_saver = []
    ddqn_scores = []
    eps_history = []
    saved_games = []
    prec = 40
    reached = 0
    reached_last_100 = 0

    agent = Agent(gamma=0.99, epsilon=1.0, lr=1 * 5e-3, n_actions=4, input_dims=[env_size[0] * env_size[1]],
                  mem_size=100000, batch_size=64, eps_min=0.01, eps_dec=5 * 1e-5, replace=100)

    if resume:
        agent.load_models()

    # Main training loop
    for i_game in tqdm(range(n_games)):
        score = 0
        avg_score = 0
        done = False
        env = Game.random(env_size, MAX_REWARD)
        observation = env.reset()
        game_sav = []
        time_step = 0
        while not done:
            time_step += 1

            action = agent.choose_action(observation)
            next_observation, reward, done = env.step(action)
            if reward == MAX_REWARD:
                reached += 1
                if i_game > (n_games - 100):
                    reached_last_100 += 1

            score += reward
            agent.store_transition(observation, action,
                                   reward, next_observation, int(done))

            agent.learn()
            observation = next_observation

            game_sav.append(next_observation)
            eps_history.append(agent.epsilon)

            ddqn_scores.append(score)

            if time_step == timeout:
                done = True

            if i_game > 0 and i_game % 10 == 0:
                agent.save_models()

            if done and i_game > 20:
                avg_score = np.mean(ddqn_scores[-10])

        score_saver.append(score)
        if i_game > 20:
            avg_score_saver.append(avg_score)
            if i_game % int(n_games / prec) == int(n_games / prec) - 1:
                print('episode: ', i_game, 'score: %.2f' % score,
                      ' average score %.2f' % avg_score,
                      'Epsilon %.3f' % agent.epsilon,
                      'Erreicht: ' + str(reached))
        saved_games.append(game_sav)



    print("")
    print(str(n_games) + " Spieldurchläufe: " + str(reached) + " mal Ziel erreicht, Quote = " + str(reached / n_games))
    print("Quote der letzten 100 Durchläufe " + str(reached_last_100 / 100))
    plt.plot(score_saver)
    plt.show()
    plt.plot(avg_score_saver)
    plt.show()
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
