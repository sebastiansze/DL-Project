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

def train(epochs=100,  env_size=(15,15), iter_timeout=60, resume=False):
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
    for epoch in tqdm(range(epochs)):
        score = 0
        avg_score = 0
        done = False
        env = Game.random(env_size, MAX_REWARD)
        observation = env.reset()
        game_sav = []
        iteration = 0
        while not done:
            iteration += 1

            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            if reward == MAX_REWARD:
                reached += 1
                if epoch > (epochs - 100):
                    reached_last_100 += 1

            score += reward
            agent.store_transition(observation, action,
                                   reward, observation_, int(done))

            agent.learn()
            observation = observation_

            game_sav.append(observation_)
            eps_history.append(agent.epsilon)

            ddqn_scores.append(score)

            if iteration == iter_timeout:
                done = True

            if epoch > 0 and epoch % 10 == 0:
                agent.save_models()

            if done and epoch > 20:
                avg_score = np.mean(ddqn_scores[-10])

        score_saver.append(score)
        if epoch > 20:
            avg_score_saver.append(avg_score)
            if epoch % int(epochs / prec) == int(epochs / prec) - 1:
                print('episode: ', epoch, 'score: %.2f' % score,
                      ' average score %.2f' % avg_score,
                      'Epsilon %.3f' % agent.epsilon,
                      'Erreicht: ' + str(reached))
        saved_games.append(game_sav)


    # What was this supposed to do? Definitely does not work like this!

    # helper = Helpers(env)
    # print("")
    # print(str(epochs) + " Spieldurchläufe: " + str(reached) + " mal Ziel erreicht, Quote = " + str(reached / epochs))
    # print("Quote der letzten 100 Durchläufe " + str(reached_last_100 / 100))
    # plt.plot(score_saver)
    # plt.show()
    # plt.plot(avg_score_saver)
    # plt.show()
    # randomgame = np.random.randint(1,20)
    # print(randomgame)
    #
    # fig, ax = plt.subplots(figsize=(env_size[1]+3, env_size[0]))
    # sns.heatmap(helper.show_reward(saved_games[-randomgame][0],elemMax=True).reshape(boardSize[0], boardSize[1]), annot=True, fmt="d")
    # plt.show()


if __name__ == '__main__':
    train()
