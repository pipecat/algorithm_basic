import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params:dict, grads:dict):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key in params.keys():
                self.v[key] = np.zeros_like(params[key])

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            self.params[key] += self.v[key]

class Adam:
    def __init__(self, lr=0.01, h=None):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key in params.keys():
                self.h[key] = np.zeros_like(params[key])

        for key in params.keys():
            self.h += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h) + 1e-7)