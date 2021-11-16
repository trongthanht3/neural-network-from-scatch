import numpy as np

class Optimizer:
    def __init__(self):
        raise NotImplementedError

    def update_params(self, layer):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, learining_rate=1.0):
        self.learning_rate = learining_rate

    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases