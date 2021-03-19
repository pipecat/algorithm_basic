import sys, os
sys.path.append(os.pardir)
import numpy as np
from deeplearning_from_scratch.net import TwoLayerNet
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_state_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]
grad_numericial = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numericial.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numericial[key]))
    print(key + ":" + str(diff))
