import numpy as np
import copy


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return self.__class__ != other.__class__ or self.x != other.x or self.y != other.y

    def distance_to(self, to, special=False):
        if special:
            return np.sqrt(np.square(self.x - to.x) + np.square(self.y - to.y))
        else:
            return np.sqrt(np.square(self.x - to.x) + np.square(self.y - to.y))


class Game:
    def __init__(self, start_pos: Point, aim_pos: Point, obstacles, board_size=(10, 10), max_reward=100, safe_dist=3):
        self.start_pos = copy.deepcopy(start_pos)
        self.player_pos = start_pos
        self.aim = aim_pos
        self.board_size = board_size
        self.obstacles = obstacles
        self.board_size = board_size
        self.MAX_REWARD = max_reward
        self.safe_dist = safe_dist
        self.reward = 0

    @staticmethod
    def random(env_size=(10, 10), max_reward=100, safe_dist=3):
        aim_pos = Point(
            np.random.randint(6, env_size[0] - 2), np.random.randint(int(env_size[1] / 2 + 2), env_size[1] - 2))
        player_pos = Point(np.random.randint(2, 4), np.random.randint(2, 5))
        num_obstacles = np.random.randint(15, 25)
        obstacles = []
        for i in range(num_obstacles):
            obstacles.append(Point(np.random.randint(1, env_size[0]), np.random.randint(1, env_size[1])))
        return Game(player_pos, aim_pos, obstacles, board_size=env_size, max_reward=max_reward)

    def reset(self):
        self.player_pos = self.start_pos
        self.reward = 0

        return self.create_map().reshape(self.board_size[0] * self.board_size[1])

    def step(self, action):
        if action == 0:
            self.player_pos.x += 1
        elif action == 1:
            self.player_pos.x -= 1
        elif action == 2:
            self.player_pos.y += 1
        elif action == 3:
            self.player_pos.y -= 1

        observ = self.create_map().reshape(self.board_size[0] * self.board_size[1])

        reward, done = self.get_reward_for_position(Point(self.board_size[0], self.board_size[1]))

        return observ, reward, done

    def create_map(self, plot=False):

        m = np.zeros(self.board_size)
        for ob in self.obstacles:
            m[ob.x, ob.y] = 0.3
        m[self.aim.x, self.aim.y] = 0.9

        if 0 < self.player_pos.x < self.board_size[0] and \
                0 < self.player_pos.y < self.board_size[1]:
            m[self.player_pos.x, self.player_pos.y] = 1

        if plot:
            plt.imshow(m, cmap='hot', interpolation='nearest')
            plt.show()
        return m

    def check_bounds(self, p: Point):
        if p.x < 0:
            return -10000, True
        elif p.y < 0:
            return -10000, True
        elif p.x >= self.board_size[0]:
            return -10000, True
        elif p.y >= self.board_size[1]:
            return -10000, True
        else:
            return 0, False

    def get_reward_for_position(self, pos: Point):
        reward = 0
        reward -= 300 * pos.distance_to(self.aim, special=True) / (self.board_size[1] + self.board_size[0])
        rew, done = self.check_bounds(pos)
        reward += rew
        for ob in self.obstacles:
            reward -= 1000 * np.exp(-(pos.distance_to(ob, special=True) * self.safe_dist))

        edge_control = 600
        if pos.x == 0 or pos.x == self.board_size[0] - 1:
            reward -= edge_control
        if pos.y == 0 or pos.y == self.board_size[1] - 1:
            reward -= edge_control
        if pos == self.aim:
            reward = self.MAX_REWARD
            done = True
        return reward, done
