import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import seaborn as sns;
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from Agent import Agent
from GameLogic import game, obstacle
from visualisation import helpers

boardSize = (15,15)   # Legt die Größe des Feldes fest

MAX_REWARD = 20000
# gameStates = np.zeros((timeSteps, boardSize[0],boardSize[1] ))

agent = Agent(gamma=0.99, epsilon=1.0, lr=1 * 5e-3, n_actions=4, input_dims=[boardSize[0] * boardSize[1]],
              mem_size=100000, batch_size=64, eps_min=0.01, eps_dec=5 * 1e-5, replace=100)
load_checkpoint = False

if load_checkpoint:
    agent.load_models()

scores, eps_hist = [], []
n_games = 100

num_obstaces = np.random.randint(15, 25)
obstacles = []
for i in range(num_obstaces):
    obstacles.append(obstacle(np.random.randint(1, boardSize[0]), np.random.randint(1, boardSize[1])))

score_saver = []
avg_score_saver = []
ddqn_scores = []
eps_history = []
savedGames = []
MAX_ITER = 60
prec = 40
reached = 0
reached_last_100 = 0
gobalenv = 0
for i in tqdm(range(n_games)):
    score = 0
    done = False
    aimPos = np.array(
        [np.random.randint(6, boardSize[0] - 2), np.random.randint(int(boardSize[1] / 2 + 2), boardSize[1] - 2)])
    playerpos = np.array([np.random.randint(2, 4), np.random.randint(2, 5)])
    env = game(playerpos, aimPos, obstacles, boardSize=boardSize,MAX_REWARD= MAX_REWARD)
    observation = env.reset()
    gobalenv = env
    game_sav = []
    iteration = 0
    while not done:
        iteration += 1

        action = agent.choose_action(observation)
        observation_, reward, done = env.step(action)
        if reward == MAX_REWARD:
            reached += 1
            if i > (n_games - 100):
                reached_last_100 += 1

        score += reward
        agent.store_transition(observation, action,
                               reward, observation_, int(done))

        agent.learn()
        observation = observation_

        game_sav.append(observation_)
        eps_history.append(agent.epsilon)

        ddqn_scores.append(score)
        if (i > 20):
            avg_score = np.mean(ddqn_scores[-10])

        if iteration == MAX_ITER:
            done = True

        if i % 10 == 0 and i > 0:
            agent.save_models()

    score_saver.append(score)
    if (i > 20):
        avg_score_saver.append(avg_score)
        if i % int(n_games / prec) == int(n_games / prec) - 1:
            print('episode: ', i, 'score: %.2f' % score,
                  ' average score %.2f' % avg_score,
                  'Epsilon %.3f' % agent.epsilon,
                  'Erreicht: ' + str(reached))
    savedGames.append(game_sav)

helper = helpers(obstacles, gobalenv,boardSize=boardSize, MAX_REWARD=100, safeDist=3 )
print("")
print(str(n_games) + " Spieldurchläufe: " + str(reached) + " mal Ziel erreicht, Quote = " + str(reached / n_games))
print("Quote der letzten 100 Durchläufe " + str(reached_last_100 / 100))
plt.plot(score_saver)
plt.show()
plt.plot(avg_score_saver)
plt.show()


randomgame = np.random.randint(1,20)
print(randomgame)

fig, ax = plt.subplots(figsize=(boardSize[1]+3,boardSize[0]))
sns.heatmap(helper.showReward(savedGames[-randomgame][0],elemMax=True).reshape(boardSize[0], boardSize[1]), annot=True, fmt="d")
plt.show()

#frame_array = helper.showhist_ani(savedGames[-randomgame])
#playVideo('output.mp4')