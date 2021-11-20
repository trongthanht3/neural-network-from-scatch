import numpy as np
from Layer import Dense, Dropout
from Loss import BinaryCrossentropy
from tqdm import tqdm
from Metrics import Accuracy
import sys

class Network:
    def __init__(self):
        self.layers = []
        self.weight = []

    def add(self, layer):
        """
            Add layer
            :param layer:
            :return:
        """
        self.layers.append(layer)

    def compile(self, loss_function, optimizer, metrics):
        """
            Set loss_function and optimizer
            :param loss_function:
            :param optimizer:
            :return:
        """
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.metrics = metrics

    def forward(self, X, training):
        output = X.copy()

        if training == True:
            for layer in self.layers:
                output = layer.forward(output)
        else:
            for layer in self.layers:
                if isinstance(layer, Dropout):
                    continue
                output = layer.forward(output)

        return output

    def backward(self, output):
        output_back = output.copy()
        for layer in reversed(self.layers):
            output_back = layer.backward(output_back)

        return output_back

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

    def fit(self, X_train, y_train, epochs=1, batch_size=None, validation_data=None):
        """
            Train model
            :param X_train:
            :param y_train:
            :param epochs:
            :param validation_data:
            :return:
        """
        samples = len(X_train)
        output = X_train

        self.metrics.init(y_train)

        train_steps = 1

        # check validation data
        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data

        if batch_size is not None:
            train_steps = len(X_train) // batch_size
            if train_steps * batch_size < len(X_train):
                train_steps += 1
            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        # Training loop
        for i in range(1, epochs + 1):
            print("\nEpoch: {}/{}".format(i, epochs))
            # init param
            loss = 0
            # output = X_train.copy()
            self.loss_function.new_pass()
            self.metrics.new_pass()
            # Iterate over steps:
            for step, _ in zip(range(train_steps), tqdm(range(train_steps))):
                # if batch_size is none, train all
                if batch_size is None:
                    batch_X = X_train
                    batch_y = y_train
                else:
                    batch_X = X_train[step * batch_size:(step + 1) * batch_size]
                    batch_y = y_train[step * batch_size:(step + 1) * batch_size]

                # forward propagation
                output = self.forward(batch_X, training=True)

                # compute loss
                data_loss = self.loss_function.caculate(output, batch_y)
                # caculate regularization penalty
                regularization_loss = 0
                for layer in self.layers:
                    if isinstance(layer, Dense):
                        regularization_loss += self.loss_function.regularization_loss(layer)
                loss = data_loss + regularization_loss

                # caculate accuracy
                if isinstance(self.loss_function, BinaryCrossentropy):
                    predictions = (output > 0.5) * 1
                else:
                    predictions = np.argmax(output, axis=1)
                accuracy = self.metrics.calculate(predictions=predictions, y=batch_y)

                # backward propagation
                loss_back = self.loss_function.backward(output, batch_y)
                loss_back = self.backward(loss_back)

                # update weight and biases by optimizer
                self.optimizer.pre_update_params()
                for layer in self.layers:
                    if isinstance(layer, Dense):
                        self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                # print log
            print('loss={:.5f} (data_loss={:.5f}, reg_loss={:.5f})'
                  .format(loss, data_loss, regularization_loss), end=' | ')
            print("Accuracy={:.5f}".format(accuracy), end=' | ')
            print("lr={:.5f}".format(self.optimizer.current_learning_rate), end=" | ")

            if validation_data is not None:
                self.loss_function.new_pass()
                self.metrics.new_pass()
                self.eval(X_val, y_val)
                # if batch_size is None:
                #     self.eval(X_val, y_val)
                # # Otherwise slice a batch
                # else:
                #     batch_X = X_val[
                #                   step * batch_size:(step + 1) * batch_size
                #                   ]
                #     batch_y = y_val[
                #               step * batch_size:(step + 1) * batch_size
                #               ]
                #     # Perform the forward pass
                #     self.eval(batch_X, batch_y)

    def eval(self, X_test, y_test):
        """
            Evaluation on test_set
            :param X_test:
            :param y_test:
            :return:
        """
        output = X_test.copy()


        # data forward to model
        output = self.forward(output, training=False)

        # compute loss
        loss = self.loss_function.caculate(output, y_test)

        # caculate accuracy
        if isinstance(self.loss_function, BinaryCrossentropy):
            predictions = (output > 0.5) * 1
        else:
            predictions = np.argmax(output, axis=1)
        accuracy = self.metrics.calculate(predictions=predictions, y=y_test)
        print("Validation: accuracy={:.5f} | loss={:.5f}".format(accuracy, loss))

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
