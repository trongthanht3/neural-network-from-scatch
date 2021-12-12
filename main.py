import sys
import argparse
import numpy as np
import MNIST_file_parser as mnist
from Network import Network
from Layer import Dense, Dropout
from Activation import ReLU, Softmax, Hardlim, Sigmoid
from Loss import CategoricalCrossentropy, MeanSquaredError, BinaryCrossentropy
from Optimizers import SGD, Adagrad, RMSprop, Adam, SteepestDecent
from Metrics import Accuracy_Categorical


def train(optimizer_name, save_model=False):
    ####################### train model
    X, y = mnist.read("training")
    y = np.ravel(y)
    Xt, yt = mnist.read("testing")
    yt = np.ravel(yt)

    print("Training set shape:", X.shape)
    print("Testing set shape:", Xt.shape)
    print("Sample train label:", y[:10])
    print("Sample test label:", yt[:10])
    sys.stdout.flush()

    model = Network()
    model.add(Dense(784, 256, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(Dense(256, 128))
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(Dense(128, 128))
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(Dense(128, 10))
    model.add(Softmax())

    if optimizer_name == "sgd":
        model.compile(
            loss_function=CategoricalCrossentropy(),
            optimizer=SGD(learining_rate=0.006, decay=1e-4, momentum=0.9),
            metrics=Accuracy_Categorical()
        )
        save_log = 'logs/logSGD.csv'

    if optimizer_name == 'adagrad':
        model.compile(
            loss_function=CategoricalCrossentropy(),
            optimizer=Adagrad(learining_rate=0.006, decay=1e-4, epsilon=1e-7),
            metrics=Accuracy_Categorical()
        )
        save_log = 'logs/logAdagrad.csv'
    if optimizer_name == 'rmsprop':
        model.compile(
            loss_function=CategoricalCrossentropy(),
            optimizer=RMSprop(decay=1e-4, epsilon=1e-7, rho=0.999),
            metrics=Accuracy_Categorical()
        )
        save_log = 'logs/logRMSprop.csv'

    if optimizer_name == 'adam':
        model.compile(
            loss_function=CategoricalCrossentropy(),
            optimizer=Adam(decay=1e-4),
            metrics=Accuracy_Categorical()
        )
        save_log = 'logs/logAdam.csv'

    # model.compile(CategoricalCrossentropy(), SteepestDecent(learining_rate=0.39, decay=0))

    model.summary()
    # model.fit(X, y, epochs=50, batch_size=128, validation_data=(Xt, yt), save_log=save_log)

    if save_model:
        model.save_model("mnist.pkl")


def test_model(mnist=mnist):
    ####################### test model
    Xt, yt = mnist.read("testing")
    yt = np.ravel(yt)

    mnist = {
        0: '0',
        1: '1',
        2: '2',
        3: '3',
        4: '4',
        5: '5',
        6: '6',
        7: '7',
        8: '8',
        9: '9'
    }

    model = Network()

    model = model.load_model("./mnist.pkl")

    result = model.predict(Xt)
    result = np.argmax(result, axis=1)
    print("Predict result: ", result[:10])
    print("True result: ", yt[:10])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("optimizer", help="choose optimzer to train",
                        type=str)
    parser.add_argument("save_model", help="True/False save model",
                        type=bool)
    args = parser.parse_args()


    print("Training model with {}.............".format(args.optimizer))
    optimizer_name = args.optimizer
    save_model = args.save_model
    train(optimizer_name, save_model)