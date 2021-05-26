import numpy as np


def softmax(X):
    c = np.max(X, axis=1, keepdims=True)
    exp = np.exp(X - c)
    s = np.sum(exp, axis=1, keepdims=True)
    return exp / s


def relu(X):
    return np.maximum(X, 0)


def d_relu(X):
    return np.where(X > 0, 1, 0)


def sigmod(X):
    X = X.astype(np.float128)
    return np.where(X > 0, 1 / (np.exp(-X) + 1), np.exp(X) / (np.exp(X) + 1))


def d_sigmod(X):
    return sigmod(X) * (1 - sigmod(X))


def d_x(X):
    return 1


def tanh(X):
    return np.where(X > 0, (np.exp(2 * X) + 1) / (np.exp(2 * X) - 1), (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X)))


def d_tanh(X):
    return 1 - np.square(tanh(X))


Activation = {'input': None, 'relu': relu, 'sigmod': sigmod, 'tanh': tanh, 'softmax': softmax}
DActivation = {'input': d_x, 'relu': d_relu, 'sigmod': d_sigmod, 'tanh': d_tanh, 'softmax': None}


class Layer():
    def __init__(self, n, activation='input'):
        self.n = n
        self.activation = Activation[activation]
        self.dactivation = DActivation[activation]
        self.act_name = activation

    def get_n(self):
        return self.n

    def get_activation_name(self):
        return self.act_name

    def set_value(self, A):
        self.A = A
