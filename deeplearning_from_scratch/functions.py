import numpy as np

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t) ** 2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def softmax(x):
    '''
    if x.ndim == 2:
        x = x - np.max(x, axis=1).reshape(x.shape[0], 1)
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(x.shape[0], 1)

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
    '''
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
