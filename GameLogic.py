import numpy as np

class game:
    def __init__(self, startPos, aimPos, obstacles, boardSize=(10,10), MAX_REWARD = 100,safeDist = 3):
        self.startPos = np.copy(startPos)
        self.playerPos = startPos
        self.aim = aim(aimPos[0], aimPos[1])
        self.boardSize = boardSize
        self.obstacles = obstacles
        self.boardSize = boardSize
        self.MAX_REWARD = MAX_REWARD
        self.safeDist = safeDist
    def reset(self):
        self.playerPos = self.startPos
        self.reward = 0

        return self.createMap().reshape(self.boardSize[0] * self.boardSize[1])

    def step(self, action):
        done = False

        if (action == 0):
            self.playerPos[0] += 1
        if (action == 1):
            self.playerPos[0] -= 1
        if (action == 2):
            self.playerPos[1] += 1
        if (action == 3):
            self.playerPos[1] -= 1

        observ = self.createMap().reshape(self.boardSize[0] * self.boardSize[1])

        reward, done = self.getRewardForField(self.playerPos[0], self.playerPos[1])

        return observ, reward, done

    def createMap(self, plot=False):

        m = np.zeros(self.boardSize)
        for ob in self.obstacles:
            m[ob.x, ob.y] = 0.3
        m[self.aim.x, self.aim.y] = 0.9

        if (self.playerPos[0] > 0 and self.playerPos[0] < self.boardSize[0] and
                self.playerPos[1] > 0 and self.playerPos[1] < self.boardSize[1]):
            m[self.playerPos[0], self.playerPos[1]] = 1

        if plot:
            plt.imshow(m, cmap='hot', interpolation='nearest')
            plt.show()
        return m

    def checkBounds(self, p):
        if (p[0] < 0):
            return -10000, True
        if (p[1] < 0):
            return -10000, True
        if (p[0] >= self.boardSize[0]):
            return -10000, True
        if (p[1] >= self.boardSize[1]):
            return -10000, True
        return 0, False

    def distance(self, a, b, special=False):
        if (special):
            return np.sqrt(np.square(a[0] - b.x) + np.square(a[1] - b.y))
        else:
            return np.sqrt(np.square(a.x - b.x) + np.square(a.y - b.y))

    def getRewardForField(self, x, y):
        done = False
        pos = [x, y]
        reward = 0
        reward -= 300 * (self.distance(pos, self.aim, special=True)) / (self.boardSize[1] + self.boardSize[0])
        rew, done = self.checkBounds(pos)
        reward += rew
        for ob in self.obstacles:
            reward -= 1000 * np.exp(-(self.distance(pos, ob, special=True) * self.safeDist))

        edgeControl = 600
        if (pos[0] == 0 or pos[0] == self.boardSize[0] - 1):
            reward -= edgeControl
        if (pos[1] == 0 or pos[1] == self.boardSize[1] - 1):
            reward -= edgeControl
        if (x == self.aim.x and y == self.aim.y):
            reward = self.MAX_REWARD
            done = True
        return reward, done


class aim:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class obstacle:
    def __init__(self, x, y):
        self.x = x
        self.y = y