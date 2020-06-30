import numpy as np
from map import print_layers


class Network:
    def __init__(self):
        self._weights = None

    def generate_agents(self, agent_count):
        return [Agent(weights=self._weights) for _ in range(agent_count)]

    def backward(self, predicted, truth):
        # TODO
        pass


class Agent:
    def __init__(self, weights):
        self._weights = weights

    def predict(self, input_map):
        """
        Do forward pass
        :return:
        """

        obstacle_map = input_map[0]
        own_current_position_coordinates = np.argwhere(input_map[1])[0]
        own_aim_position_coordinates = np.argwhere(input_map[2])[0]
        others_current_positions_map = input_map[3]

        print_layers(obstacle_map)
        print(own_current_position_coordinates)
        print(own_aim_position_coordinates)
        print_layers(others_current_positions_map)
        print()
        print('Hier steht ein "input()". Deshalb gehts nicht weiter...')
        input()

        # TODO
        y = [1, 0, 0, 0, 0]
        np.random.shuffle(y)
        return y
