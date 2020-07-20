import numpy as np
import copy
from typing import List, Tuple


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return self.__class__ != other.__class__ or self.x != other.x or self.y != other.y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def __str__(self):
        return f"[{self.x}, {self.y}]"

    def distance_to(self, to):
        return np.sqrt(np.square(self.x - to.x) + np.square(self.y - to.y))

    def to_numpy(self):
        return np.array([self.x, self.y])


class Player:
    def __init__(self, start: Point, aim: Point):
        self.start = start
        self.position = copy.deepcopy(start)
        self.previous_position = copy.deepcopy(start)
        self.aim = aim

    def reset(self):
        self.position = copy.deepcopy(self.start)
        self.previous_position = copy.deepcopy(self.start)

    def move(self, action):
        """
        :param action: Moves player according to action
        """
        self.previous_position = copy.deepcopy(self.position)
        if action == 0:
            self.position.x += 1
        elif action == 1:
            self.position.x -= 1
        elif action == 2:
            self.position.y += 1
        elif action == 3:
            self.position.y -= 1

    def did_collide_with(self, other: 'Player'):
        # True if players switched places and therefore ran front to front into each other
        return other.position == self.previous_position and self.position == other.previous_position


class Game:
    def __init__(self, obstacles, players: List[Player] = None, board_size=(10, 10), max_reward=100, safe_dist=3,
                 view_size=(2, 2, 2, 2), view_reduced=False):
        self.players = [] if players is None else players
        self.board_size = board_size
        self.obstacles = obstacles
        self.MAX_REWARD = max_reward
        self.safe_dist = safe_dist
        self.view_size = view_size
        self.viewTo1D = (self.view_size[0] + self.view_size[1] + 1) * (self.view_size[2] + self.view_size[3] + 1)
        self.view_reduced = view_reduced

    def add_player(self, start: Point = None, aim: Point = None, min_distance = None):
        """
        :param min_distance: minimum distance between start and aim
        :param start: a Point with the start coordinates, will be generated randomly if None
        :param aim: a Point with aim coordinates, will be generated randomly if None
        """
        occupied_starts = [player.start for player in self.players]
        occupied_aims = [player.aim for player in self.players]

        if start is None:
            valid = False
            while not valid:
                start = Point(np.random.randint(0, self.board_size[0]), np.random.randint(0, self.board_size[1]))
                valid = (start not in occupied_starts and
                         start not in self.obstacles and
                         start not in occupied_aims)

        if aim is None:
            valid = False
            while not valid:
                aim = Point(np.random.randint(0, self.board_size[0]),
                            np.random.randint(0, self.board_size[1]))
                valid = (aim not in occupied_aims and
                         aim not in self.obstacles and
                         aim not in occupied_starts and
                         aim != start)
                if min_distance is not None:
                    valid = valid and start.distance_to(aim) >= min_distance

        self.players.append(Player(start, aim))

    @staticmethod
    def random(board_size=(10, 10), player_count=1, max_reward=100, safe_dist=3):
        num_obstacles = np.random.randint(15, 25)
        obstacles = []
        for i in range(num_obstacles):
            obstacles.append(Point(np.random.randint(1, board_size[0]), np.random.randint(1, board_size[1])))
        game = Game(obstacles, board_size=board_size, max_reward=max_reward, safe_dist=3)
        for i_player in range(player_count):
            game.add_player()
        return game

    def reset(self):
        for player in self.players:
            player.reset()
        observations = []
        for player_id in range(len(self.players)):
            observations.append(self.create_map_for_player_id(player_id).flatten())
        return observations

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool]]:
        """
        :param actions: Mapping from player_id to action for that player
        :return: mapping from player_id to observation, reward, final
        """
        if len(actions) != len(self.players):
            raise RuntimeError(f"Length of actions is not equal to number of players! Expected: {len(self.players)}"
                               f" but got {len(actions)}")

        observations = []
        rewards = []
        in_final_state = []
        # First move all players
        for player_id, action in enumerate(actions):
            if action is not None:
                self.players[player_id].move(action)

        # Then calculate map and reward for each player
        for player_id, action in enumerate(actions):
            reward, final_state = self.get_reward_for_player_id(player_id)
            observations.append(self.create_map_for_player_id(player_id).flatten())
            rewards.append(reward)
            in_final_state.append(final_state)

        return observations, rewards, in_final_state

    def create_map_for_player_id(self, player_id: int):
        m = np.zeros(self.board_size)

        playerSav = (0, 0)
        aimSav = (0,0)
        for ob in self.obstacles:
            m[ob.x, ob.y] = 0.25
        for id_, player in enumerate(self.players):

            if id_ == player_id:
                m[player.aim.x, player.aim.y] = 0.75
                if 0 <= player.position.x < self.board_size[0] and 0 <= player.position.y < self.board_size[1]:
                    m[player.position.x, player.position.y] = 1
                    playerSav = (np.copy(player.position.x), np.copy(player.position.y))
                    aimSav = (player.aim.x, player.aim.y)
                    #print("Saved at ", str(playerSav[0]) + " " + str(playerSav[1]))
            else:
                # Only set position if it is not own position
                if 0 <= player.position.x < self.board_size[0] and 0 <= player.position.y < self.board_size[1]\
                        and m[player.position.x, player.position.y] != 0.75:
                    m[player.position.x, player.position.y] = 0.5
                # Show aims of other agents as an obstacle of this agent
                m[player.aim.x, player.aim.y] = 0.25
        #print("Viewed  at ", str(playerSav[0]) + " " + str(playerSav[1]))
        observation = self.getView(m, self.view_size, (playerSav[0], playerSav[1]))

        #print(observation)
        if self.view_reduced:

            data = np.array([playerSav[0] / self.board_size[0],
                             playerSav[1] / self.board_size[1],
                             aimSav[0] / self.board_size[0],
                             aimSav[1] / self.board_size[1]])
            m = np.concatenate((observation, data), axis=None)

        return m

    def getView(self, board, view, pp):
        # board[pp[0], pp[1]] = 1
        out = np.zeros((1 + view[0] + view[1], 1 + view[2] + view[3]))
        xMin = pp[0] - view[0]
        xMax = pp[0] + view[1]
        yMin = pp[1] - view[2]
        yMax = pp[1] + view[3]

        if xMin < 0:
            xMinOut = 0
        else:
            xMinOut = xMin

        if yMin < 0:
            yMinOut = 0
        else:
            yMinOut = yMin

        # --- Offsets ---

        if xMin < 0:
            xMinOffset = -xMin
        else:
            xMinOffset = 0

        if xMax >= board.shape[0]:
            xMaxOffset = out.shape[0] + board.shape[0] - xMax - 1
        else:
            xMaxOffset = out.shape[0]

        if yMin < 0:
            yMinOffset = -yMin
        else:
            yMinOffset = 0

        if yMax >= board.shape[1]:
            yMaxOffset = out.shape[1] + board.shape[1] - yMax - 1
        else:
            yMaxOffset = out.shape[1]
        # print("PlayerPos = " + str(self.playerPos[0]) + "," + str(self.playerPos[1]))
        out[xMinOffset:xMaxOffset, yMinOffset:yMaxOffset] = board[xMinOut:xMax + 1, yMinOut:yMax + 1]
        # print(out)
        return out

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

    def get_reward_for_player_id(self, player_id: any):
        player = self.players[player_id]
        reward, done = self.check_bounds(player.position)
        reward -= 300 * player.position.distance_to(player.aim) / (self.board_size[1] + self.board_size[0])

        # reduce reward if we get close to / collide with an obstacle
        for ob in self.obstacles:
            reward -= 1000 * np.exp(-(player.position.distance_to(ob) * self.safe_dist))
            if ob == player.position:
                done = True

        # reduce reward if we get close to / collide with another player
        for player_b_id, player_b in enumerate(self.players):
            if player_b_id != player_id:
                reward -= 1000 * np.exp(-(player.position.distance_to(player_b.position) * self.safe_dist))
                # If players ended up on same field or crashed during move terminate them
                if player.position == player_b.position or player.did_collide_with(player_b):
                    done = True

        # Penalize being close to the board borders
        edge_control = 600
        if player.position.x == 0 or player.position.x == self.board_size[0] - 1:
            reward -= edge_control
        if player.position.y == 0 or player.position.y == self.board_size[1] - 1:
            reward -= edge_control
        if player.position == player.aim:
            reward = self.MAX_REWARD
            done = True
        return reward, done
