import numpy as np
import random

random.seed(1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, weights: [], bias=0):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inp):
        total = np.dot(self.weights, inp) + self.bias
        return sigmoid(total)


class NeuralNetwork:
    def __init__(self):
        self.h1 = Neuron([random.random(), random.random()])
        self.h2 = Neuron([random.random(), random.random()])
        self.o1 = Neuron([random.random(), random.random()])

    def feedforward(self, inp):
        out1 = self.h1.feedforward(inp)
        out2 = self.h2.feedforward(inp)

        out = self.o1.feedforward([out1, out2])

        return out


network = NeuralNetwork()

print(network.feedforward([2, 3]))
