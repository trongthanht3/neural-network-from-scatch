import numpy as np


class Loss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self, dvalues, y_true):
        raise NotImplementedError

    def regularization_loss(self, layer):
        """
            Regularization methods are those which reduce generalization error.
            Large weights might indicate that a neuron is attempting to memorize a data
            element; generally, it is believed that it would be better to have
            many neurons contributing to a model’s output, rather than a select few.
            :param layer:
            :return:
        """
        regularization_loss = 0

        # l1 regularization - weights
        # caculate only when factor greater than 0
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        # l2 regularization - weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

        # l1 regularization - biases
        # caculate only when factor greater than 0
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        # l2 regularization - biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss

    def caculate(self, outputs, y):
        sample_losses = self.forward(outputs, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class CategoricalCrossentropy(Loss):
    """
        Categorical cross-entropy is explicitly used to compare
        a “ground-truth” probability (y or “targets”) and some
        predicted distribution (y-hat or “predictions”),
        so it makes sense to use cross-entropy here.
    """

    def forward(self, y_pred, y_true):
        # numbers of samples in a batch
        samples = len(y_pred)

        # clip data to prevent division by 0
        # clip both sides to not to drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # probabilities for target
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        # print(dvalues)
        samples = len(dvalues)

        # use first sample to count
        labels = len(dvalues[0])

        # turn into one-hot if sparse
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # caculate gradient
        self.dinputs = -y_true / dvalues
        # noramlize
        self.dinputs = self.dinputs / samples

        return self.dinputs

class MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        return np.mean(np.power(y_true - y_pred, 2))


class MeanSquaredErrorPrime(Loss):
    def forward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.size;
