import numpy as np
from Network import Network
from Layer import Dense, Dropout
from Activation import ReLU, Softmax, Hardlim, Sigmoid
from Loss import CategoricalCrossentropy, MeanSquaredError, BinaryCrossentropy
from Optimizers import SGD, Adagrad, RMSprop, Adam, SteepestDecent
from Metrics import Accuracy_Categorical
import sys
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data
import matplotlib.pyplot as plt
import MNIST_file_parser as mnist

################## classification
# X,y = spiral_data(10000, 3)
# Xt, yt = spiral_data(100, 3)
# keys = np.array(range(X.shape[0]))
# np.random.shuffle(keys)
# X = X[keys]
# y = y[keys]

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
#
#
# model.compile(CategoricalCrossentropy(), SGD(learining_rate=1, decay=1e-3, momentum=0.9))
# model.compile(CategoricalCrossentropy(), Adagrad(learining_rate=1, decay=1e-4, epsilon=1e-7))
# model.compile(CategoricalCrossentropy(), RMSprop(learining_rate=0.02, decay=1e-5, epsilon=1e-7, rho=0.999))
model.compile(
    loss_function=CategoricalCrossentropy(),
    optimizer=Adam(decay=1e-4),
    metrics=Accuracy_Categorical()
)
# # model.compile(CategoricalCrossentropy(), SteepestDecent(learining_rate=0.39, decay=0))
#
model.fit(X, y, epochs=100, batch_size=128, validation_data=(Xt, yt))

