import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

from layer import *


def one_hot_encoder(X, catagory=None):
    if(catagory == None):
        catagory = np.unique(X)
    ohc = np.zeros((len(X), len(catagory)))
    for i in range(len(X)):
        t = (catagory == X[i])
        pos = np.argmax(t)
        ohc[i, pos] = 1
    return ohc, catagory


class NeuralNetwork():
    def __init__(self, learning_rate=0.0001):
        self.lr = learning_rate
        self.layers = []
        self.W = []
        self.B = []

    def add(self, layer):
        if self.layers == []:
            self.layers.append(layer)
        else:
            w = np.random.randn(self.layers[-1].get_n(), layer.get_n())
            b = np.zeros([1, layer.get_n()])
            self.W.append(w)
            self.B.append(b)
            self.layers.append(layer)

    def load(self, layer):
        self.layers.append(layer)

    def summary(self):
        print('Neural network:')
        print(f'Input shape : {0, self.layers[0].get_n()}')
        for i in range(1, len(self.layers) - 1):
            print(f'Layer {i} : shape : (0, {self.layers[i].get_n()}), params : {self.W[i - 1].size + self.B[i - 1].size}, activation="{self.layers[i].get_activation_name()}"')

        print(f'Output shape : {0, self.layers[-1].get_n()}, activation = "{self.layers[-1].get_activation_name()}"')

    def forward(self, X):
        self.layers[0].set_value(X)
        for i in range(0, len(self.W)):
            Z = self.layers[i].A.dot(self.W[i]) + self.B[i]
            a = self.layers[i + 1].activation(Z)
            self.layers[i + 1].set_value(a)

    def loss(self, X, y):
        self.forward(X)
        res = -np.log(self.layers[-1].A[np.arange(len(X)), y])
        return np.sum(res)

    def back_prop(self, y):
        dW = []
        dB = []

        dZ = self.layers[-1].A - y

        for i in reversed(range(0, len(self.W))):

            dw = self.layers[i].A.T.dot(dZ)
            dW.append(dw)

            db = np.sum(dZ, axis=0, keepdims=True)
            dB.append(db)

            dA = (dZ).dot(self.W[i].T)

            dZ = dA * self.layers[i].dactivation(self.layers[i].A)

        return dW, dB

    def update(self, dW, dB):
        for i in range(len(self.W)):
            self.W[i] -= self.lr * dW[i]
            self.B[i] -= self.lr * dB[i]

    def fit(self, X, y, epochs, batch_size=1, validation_data=None):
        self.train = []
        self.val = []

        if len(self.layers) < 2:
            print("Haven't create at least 2 layers for model")
        else:
            m = len(y)
            ohc_y, catal = one_hot_encoder(y)
            batch_num = int(np.ceil(len(y) / batch_size))

            for i in range(epochs):
                print(f'Epoch : {i + 1} / {epochs}')
                print('[', end='')
                num = 1
                for j in range(batch_num):
                    while (j / batch_num) / num >= 0.025:
                        num += 1
                        print('-', end='')

                    index = np.random.randint(batch_num)

                    start = index * batch_size
                    end = (index + 1) * batch_size

                    X_t = X[start: end]
                    y_t = ohc_y[start: end]

                    self.forward(X_t)
                    dW, dB = self.back_prop(y_t)
                    self.update(dW[::-1], dB[::-1])

                print(']', end=' ')

                train_acc = self.valuate(X, y)
                self.train.append(train_acc)
                if (validation_data != None):
                    val_acc = self.valuate(validation_data[0], validation_data[1])
                    self.val.append(val_acc)
                    print(f'train_accuracy: {self.train[-1]} | val_accuracy: {self.val[-1]}')

                else:
                    print(f'loss_train: {self.train[-1]}')

    def plot(self):
        plt.style.use('seaborn-whitegrid')
        plt.figure(figsize=(12, 10))
        plt.plot(self.train, c='orange', label='Training')
        plt.plot(self.val, c='red', label='Valuation')
        plt.legend(loc='best', fontsize=16)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.show()

    def predict(self, X, return_prob=False):
        self.forward(X)
        prob = self.layers[-1].A
        if(return_prob == False):
            return np.argmax(prob, axis=1)
        else:
            return np.argmax(prob, axis=1), prob
        return Z

    def valuate(self, X, y):
        pred, prob = self.predict(X, return_prob=True)
        accuracy = np.sum((y == pred)) / len(y)
        return accuracy

    def save(self, filename):
        name, ext = filename.split('.')
        data = {'lr': self.lr}
        layer = {}
        i = 0
        for l in self.layers:
            layer['l' + str(i)] = {'neurons': l.get_n(), 'activation': l.get_activation_name()}
            i += 1
        data['layer'] = layer

        with open(name + '.npy', 'wb') as f:
            for w in self.W:
                np.save(f, w)
            for b in self.B:
                np.save(f, b)

        with open(filename, 'w') as f:
            json.dump(data, f)


def load_model(filename):
    name, ext = filename.split('.')
    with open(filename, 'r') as f:
        data = json.load(f)
        model = NeuralNetwork(data['lr'])
        layers = data['layer']
        for id, layer in layers.items():
            model.load(Layer(layer['neurons'], activation=layer['activation']))

    with open(name + '.npy', 'rb') as f:
        for i in range(len(layers) - 1):
            w = np.load(f)
            model.W.append(w)

        for i in range(len(layers) - 1):
            b = np.load(f)
            model.B.append(b)

    return model
