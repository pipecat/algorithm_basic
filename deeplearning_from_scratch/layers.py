import sys, os
sys.path.append(os.pardir)
import numpy as np

class AddLayer:

    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y

        return self.x + self.y

    def backward(self, dout):
        dx = dout
        dy = dout

        return dx, dy

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
    
        return x * y
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = ( x <= 0 )
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        out = self.y

        return out
    
    def backward(self, dy):
        dx = dy * self.y * (1.0 - self.y)
        
        return dx

class Affine:
    
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        self.dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx
