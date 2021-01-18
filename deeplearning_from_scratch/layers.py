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