import numpy as np
from Network import Network
from Layer import Dense
from Activation import ReLU, Softmax, Hardlim
from Loss import CategoricalCrossentropy, MeanSquaredError
from Optimizers import SGD, Adagrad, RMSprop, Adam, SteepestDecent
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data
import matplotlib.pyplot as plt

X, y = spiral_data(samples=100, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)
# X, y = vertical_data(samples=100, classes=3)
# plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap='brg')
# plt.show()


model = Network()
model.add(Dense(n_inputs=2, n_neurons=64))
model.add(ReLU())
model.add(Dense(n_inputs=64, n_neurons=3))
model.add(Softmax())
#
#
# model.use(CategoricalCrossentropy(), SGD(learining_rate=1, decay=1e-3, momentum=0.9))
# model.use(CategoricalCrossentropy(), Adagrad(learining_rate=1, decay=1e-4, epsilon=1e-7))
# model.use(CategoricalCrossentropy(), RMSprop(learining_rate=0.02, decay=1e-5, epsilon=1e-7, rho=0.999))
model.use(CategoricalCrossentropy(), Adam(learining_rate=0.05, decay=5e-7))
# model.use(CategoricalCrossentropy(), SteepestDecent(learining_rate=0.39, decay=0))

model.fit(X, y, 10000, 0)

model.eval(X_test, y_test)
# model.get_weight()