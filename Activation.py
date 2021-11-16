import numpy as np
from Layer import Layer


class ReLU(Layer):
    """
        The ReLU in this code is a loop where we’re checking
        if the current value is greater than 0. If it is,
        we’re appending it to the output list, and if it’s not,
        we’re appending 0.
    """
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, self.inputs)
        return self.outputs

    def backward(self, dvalues, learning_rate):
        self.dinputs = dvalues.copy()

        # zero gradient
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs

class Sigmoid(Layer):
    """
        This function returns a value in the range of 0
        for negative infinity, through 0.5 for the
        input of 0, and to 1 for positive infinity.
    """
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = 1/(1+np.exp(-1*self.inputs))
        return self.outputs

class Softmax(Layer):
    """
    This distribution returned by the softmax activation
    function represents confidence scores for each class
    and will add up to 1
    """
    def forward(self, inputs):
        self.inputs = inputs
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)
        self.outputs = probabilities
        return self.outputs

    def backward(self, dvalues, learning_rate):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.outputs, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)
        return self.dinputs

class Hardlim(Layer):
    def forward(self, inputs):
        return (inputs > 0) * 1.0


# if __name__ == '__main__':
#     relu = Softmax()
#     input = np.array([[1, 2, 3]])
#     out = relu.forward(input)
#     print(out)