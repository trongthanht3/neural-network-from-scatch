import numpy as np


class Optimizer:
    def __init__(self):
        raise NotImplementedError

    def update_params(self, layer):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, learining_rate=1.0, decay=0., momentum=None):
        self.learning_rate = learining_rate
        self.current_learning_rate = learining_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """
            Update params
            :param layer:
            :return:
        """
        if self.momentum:
            # if layer does not contain momentum, create them
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # same with biases
                layer.bias_momentums = np.zeros_like(layer.biases)

            # build weight updates with momentum
            # take previous updates multipliled by retain factor
            # and update with current gradients
            weight_updates = self.momentum * layer.weight_momentums \
                             - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            # build biases
            bias_updates = self.momentum * layer.bias_momentums \
                           - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            # vanila SGD
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1


class Adagrad(Optimizer):
    def __init__(self, learining_rate=1.0, decay=0., epsilon=1e-7):
        self.learning_rate = learining_rate
        self.current_learning_rate = learining_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """
            Update params
            :param layer:
            :return:
        """
        # if layer does not contain cache
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # update cache with squared current gradients
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        # vanila SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights \
                         / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases \
                        / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class RMSprop(Optimizer):
    def __init__(self, learining_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learining_rate
        self.current_learning_rate = learining_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """
            Update params
            :param layer:
            :return:
        """
        # if layer does not contain cache
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2

        # vanila SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights \
                         / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases \
                        / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Adam(Optimizer):
    def __init__(self, learining_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learining_rate
        self.current_learning_rate = learining_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2  # rho now become beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """
            Update params
            :param layer:
            :return:
        """
        # if layer does not contain cache
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.bias_momentums = np.zeros_like(layer.biases)

        # momentum
        # update momentum with current gradient
        layer.weight_momentums = self.beta_1 * layer.weight_momentums \
                                 + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums \
                               + (1 - self.beta_1) * layer.dbiases
        # get correct momentum, iter = 0, we start from 1
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # cache
        # update cache with current gradient
        layer.weight_cache = self.beta_2 * layer.weight_cache \
                             + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache \
                           + (1 - self.beta_2) * layer.dbiases ** 2
        # get correct cache, iter = 0, we start from 1
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # vanila SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (
                np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (
                np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class SteepestDecent(Optimizer):
    def __init__(self, learining_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learining_rate
        self.current_learning_rate = learining_rate
        self.decay = decay
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        # update weight
        layer.weights = layer.weights - self.current_learning_rate * layer.dweights
        layer.biases = layer.biases - self.current_learning_rate * layer.biases

    def post_update_params(self):
        self.iterations += 1
