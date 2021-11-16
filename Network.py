import numpy as np
from Metrics import accuracy_score


class Network:
    def __init__(self):
        self.layers = []
        self.weight = []
        self.loss_function = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss_function):
        self.loss_function = loss_function

    def predict(self, input_data):
        result = []
        output = input_data
        # run network over all samples
        for layer in self.layers:
            output = layer.forward(inputs=output)


    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)
        output = x_train
        # Training loop
        for i in range(epochs):
            loss = 0
            output = x_train.copy()
            for layer in self.layers:
                output = layer.forward(inputs=output)
            # output now is activation_output
            # print(output)

            # compute loss
            # if self.loss_function != None:
            loss = self.loss_function.caculate(output, y_train)
            print("Loss:", loss)
            print(output.shape, y_train.shape)

            loss_back = self.loss_function.backward(output, y_train)
            # print(loss_back)
            for layer in reversed(self.layers):
                loss_back = layer.backward(loss_back, learning_rate)
            # calculate average error on all samples
            # loss /= samples
            print('Epoch: {}/{} | loss={:.5f}'.format(i + 1, epochs, loss), end=' | ')

            # caculate accuracy
            predictions = np.argmax(output, axis=1)
            accuracy = accuracy_score(y_pred=predictions, y_true=y_train)
            print("Accuracy: {}".format(accuracy))

    def get_weight(self):
        for layer in self.layers:
            try:
                print(layer.weights)
                print(layer.biases)
            except:
                pass
