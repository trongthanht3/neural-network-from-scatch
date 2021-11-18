import numpy as np
from Layer import Dense
from Metrics import accuracy_score


class Network:
    def __init__(self):
        self.layers = []
        self.weight = []
        self.loss_function = None

    def add(self, layer):
        """
            Add layer
            :param layer:
            :return:
        """
        self.layers.append(layer)

    def use(self, loss_function, optimizer):
        """
            Set loss_function and optimizer
            :param loss_function:
            :param optimizer:
            :return:
        """
        self.loss_function = loss_function
        self.optimizer = optimizer

    def predict(self, input_data):
        """
            Predict
            :param input_data:
            :return:
        """
        result = []
        output = input_data
        # run network over all samples
        for layer in self.layers:
            output = layer.forward(inputs=output)

    def fit(self, x_train, y_train, epochs, learning_rate):
        """
            Train model
            :param x_train:
            :param y_train:
            :param epochs:
            :param learning_rate:
            :return:
        """
        samples = len(x_train)
        output = x_train
        # Training loop
        for i in range(epochs):
            loss = 0
            output = x_train.copy()

            # forward propagation
            for layer in self.layers:
                output = layer.forward(inputs=output)
            # output now is activation_output
            # print(output)

            # compute loss
            # if self.loss_function != None:
            loss = self.loss_function.caculate(output, y_train)
            # print("Loss:", loss)
            # print(output.shape, y_train.shape)

            # backward propagation
            loss_back = self.loss_function.backward(output, y_train)
            # print(loss_back)
            for layer in reversed(self.layers):
                loss_back = layer.backward(loss_back, learning_rate)

            # update weight and biases by optimizer
            self.optimizer.pre_update_params()
            for layer in self.layers:
                if isinstance(layer, Dense):
                    self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            # calculate average error on all samples
            # loss /= samples
            print('Epoch: {}/{} | loss={:.5f}'.format(i + 1, epochs, loss), end=' | ')



            # caculate accuracy
            predictions = np.argmax(output, axis=1)
            accuracy = accuracy_score(y_pred=predictions, y_true=y_train)
            print("Accuracy={}".format(accuracy), end=' | ')
            print("lr={}".format(self.optimizer.current_learning_rate))

    def eval(self, X_test, y_test):
        """
            Evaluation on test_set
            :param X_test:
            :param y_test:
            :return:
        """
        output = X_test.copy()

        # data forward to model
        for layer in self.layers:
            output = layer.forward(inputs=output)

        # compute loss
        loss = self.loss_function.caculate(output, y_test)

        # caculate accuracy
        predictions = np.argmax(output, axis=1)
        accuracy = accuracy_score(y_pred=predictions, y_true=y_test)
        print("Validation: accuracy={} | loss={}".format(accuracy, loss))

    def get_weight(self):
        """
            Show weights in all layer
            :return:
        """
        for layer in self.layers:
            try:
                print(layer.weights)
                print(layer.biases)
            except:
                pass
