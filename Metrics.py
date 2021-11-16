import numpy as np

def accuracy_score(y_true, y_pred):
    """
        Caculate accuracy
        :param y_true:
        :param y_pred:
        :return:
    """
    if len(y_true.shape) == 2:
        y = np.argmax(y_true, axis=1)
    accuracy = np.mean(y_pred==y_true)
    return accuracy