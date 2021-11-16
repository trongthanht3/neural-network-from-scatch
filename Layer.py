import numpy as np


class Layer:
    def __init__(self):
        self.inputs = None
        self.outputs = None

    # computes the output Y of a layer for a given input X
    def forward(self, inputs):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward(self, dvalues, learning_rate):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, n_inputs, n_neurons):
        """
            Init layer
            :param n_inputs: numbers of feature
            :param n_neurons: number of neuron
        """
        self.weights = 0.01 * np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(self.inputs, self.weights) + self.biases
        return self.outputs

    def backward(self, dvalues, learning_rate):
        # gradients on param
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs

