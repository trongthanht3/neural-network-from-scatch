import numpy as np


class Layer:
    def __init__(self):
        self.inputs = None
        self.outputs = None

    # computes the output Y of a layer for a given input X
    def forward(self, inputs):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward(self, dvalues):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        """
            Init layer
            :param n_inputs: numbers of feature
            :param n_neurons: number of neuron
        """
        # init random weights and biases
        self.weights = 0.01 * np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # set regularization
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(self.inputs, self.weights) + self.biases
        return self.outputs

    def backward(self, dvalues):
        # gradients on param
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # gradient on regularization
        # l1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # l2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        # l1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # l2 on weights
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs

class Dropout(Layer):
    # init
    def __init__(self, rate):
        # store rate, invert it to dropout
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = inputs

        # generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate

        # apply mask to output values
        self.outputs = inputs * self.binary_mask

        return self.outputs

    def backward(self, dvalues):
        # gradient on values
        self.dinputs = dvalues * self.binary_mask

        return self.dinputs