import numpy as np
from Network import Network
from Layer import Dense
from Activation import ReLU, Softmax, Hardlim
from Loss import CategoricalCrossentropy, MeanSquaredError
from Optimizers import SGD
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data
import matplotlib.pyplot as plt

X, y = spiral_data(samples=100, classes=3)
# X, y = vertical_data(samples=100, classes=3)
# plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap='brg')
# plt.show()


model = Network()
model.add(Dense(n_inputs=2, n_neurons=3))
model.add(ReLU())
model.add(Dense(n_inputs=3, n_neurons=3))
model.add(Softmax())
#
#
model.use(CategoricalCrossentropy(), SGD())
model.fit(X, y, 1, 0)

model.get_weight()