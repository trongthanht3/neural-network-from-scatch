import copy
import numpy as np
import pandas as pd
from Layer import Dense, Dropout
from Loss import BinaryCrossentropy
from tqdm import tqdm
import sys
import pickle
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report


class Network:
    def __init__(self):
        self.layers = []
        self.weight = []
        self.log = None

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

    def summary(self):
        layer_queue = self.layers.copy()[::-1]
        layer_queue_name = [type(i).__name__ for i in self.layers.copy()[::-1]]
        print(layer_queue_name)
        counted = []
        last_neuron = 0
        total_params = 0

        print()
        print('Model: "{}"'.format(type(self).__name__))
        print(u"────────────────────────────────────────────────────────────────")
        print("Layer (type)                 Output Shape              Param #   ")
        print("=================================================================")

        while layer_queue:
            layer_pop = layer_queue.pop()
            layer_pop_name = layer_queue_name.pop()
            if counted.count(layer_pop_name) > 0:
                l_str = "{} ({})".format(str(layer_pop_name).lower(), layer_pop_name)
                print("{:<29}".format(l_str), end='')
                if isinstance(layer_pop, Dense):
                    l_oshape = "({}, {})".format(None, layer_pop.n_neurons)
                    last_neuron = layer_pop.n_neurons
                    l_params = (layer_pop.n_inputs+1) * layer_pop.n_neurons
                else:
                    l_oshape = "({}, {})".format(None, last_neuron)
                    l_params = 0
                print("{:<26}".format(l_oshape), end='')
                print("{:<10}".format(str(l_params)))
                total_params += l_params
            else:
                l_str = "{} ({})".format(str(layer_pop_name).lower(), layer_pop_name)
                print("{:<29}".format(l_str), end='')
                if isinstance(layer_pop, Dense):
                    l_oshape = "({}, {})".format(None, layer_pop.n_neurons)
                    last_neuron = layer_pop.n_neurons
                    l_params = (layer_pop.n_inputs+1) * layer_pop.n_neurons
                else:
                    l_oshape = "({}, {})".format(None, last_neuron)
                    l_params = 0
                print("{:<26}".format(l_oshape), end='')
                print("{:<10}".format(str(l_params)))
                total_params += l_params
            counted.append(layer_pop_name)
            print('-----------------------------------------------------------------')
        del counted
        print("=================================================================")
        print("Total params:", total_params)
        print('-----------------------------------------------------------------')

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

    def fit(self, X_train, y_train, epochs=1, batch_size=None, validation_data=None,
            save_log=None):
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
        if save_log:
            self.log = pd.DataFrame(columns=['run', 'epoch', 'epoch_accuracy',
                                             'epoch_precision', 'epoch_recall',
                                             'epoch_f1', 'epoch_loss'])

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
            print("\nEpoch: {}/{}".format(i, epochs), flush=True)
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
                  .format(loss, data_loss, regularization_loss), end=' | ', flush=True)
            print("Accuracy={:.5f}".format(accuracy), end=' | ', flush=True)
            print("lr={:.5f}".format(self.optimizer.current_learning_rate), end=" | ", flush=True)
            sys.stdout.flush()

            if validation_data is not None:
                _, accuracy, precision, recall, f1 = self.eval(X_val, y_val, verbose=False)
                log_step = pd.Series(index=['run', 'epoch', 'epoch_accuracy',
                                            'epoch_precision', 'epoch_recall',
                                            'epoch_f1', 'epoch_loss'])
                log_step['run'] = type(self.optimizer).__name__ + '/validation'
                log_step['epoch'] = i
                log_step['epoch_accuracy'] = accuracy
                log_step['epoch_precision'] = precision
                log_step['epoch_recall'] = recall
                log_step['epoch_f1'] = f1
                log_step['epoch_loss'] = loss
                self.log = self.log.append(log_step, ignore_index=True)

        if save_log:
            self.log.to_csv(save_log, index=False)
            print("save loggggggggggggggggggggggggg")

        self.eval(X_val, y_val, verbose=True)

    def eval(self, X_test, y_test, verbose=False):
        """
            Evaluation on test_set
            :param X_test:
            :param y_test:
            :return:
        """
        self.loss_function.new_pass()
        self.metrics.new_pass()

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
        precision = precision_score(y_true=y_test, y_pred=predictions, average='macro')
        recall = recall_score(y_true=y_test, y_pred=predictions, average='macro')
        f1 = f1_score(y_true=y_test, y_pred=predictions, average='macro')
        print("""Validation: Accuracy={:.5f} | loss={:.5f}"""
              .format(accuracy, loss, ), flush=True)
        if verbose:
            print("""Precision={} | Recall={} | F1={}"""
                  .format(precision,
                          recall,
                          f1), flush=True)
            print(classification_report(y_true=y_test, y_pred=predictions))
        return predictions, accuracy, precision, recall, f1

    def predict(self, input_data):
        """
            Predict
            :param input_data:
            :return:
        """
        result = []
        output = input_data
        # run network over all samples
        output = self.forward(output, training=False)

        return np.vstack(output)

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

    def save_model(self, path):
        model = copy.deepcopy(self)

        # remove train value
        model.loss_function.new_pass()
        model.metrics.new_pass()

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            temp = pickle.load(f)
        return temp
