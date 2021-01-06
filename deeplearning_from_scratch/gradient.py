import numpy as np

def _numerical_gradient_no_batch(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for i, x in enumerate(X):
            grad[i] = _numerical_gradient_no_batch(f, x)

        return grad

def gradient_descent(f, int_x, lr=0.01, step_num=100):
    x = int_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        
        x -= lr * grad
        print(x)

    return x

def function_1(x):
    return x**2

def function_2(x):
    return x[0]**2 + x[1]**2

print(gradient_descent(function_2, np.array([-3.,4.]).astype('float64'), lr=0.1, step_num=100))
