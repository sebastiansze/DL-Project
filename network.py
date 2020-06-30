import numpy as np


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
        # TODO
        y = [1, 0, 0, 0, 0]
        np.random.shuffle(y)
        return y
