import numpy as np
from random import random, seed

seed(1)


def sigmoid(x, der=False):
    if der:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, weight, bias=0):
        self.weight = weight
        self.bias = bias

    def feedforward(self, inp):
        total = np.dot(self.weight, inp) + self.bias
        return sigmoid(total)

    def __str__(self):
        return f'{self.weight}'

    def __repr__(self):
        return f'w: {self.weight}'


class NeuralNetwork:
    def __init__(self, input_count, hidden_count, out_count):
        input_layer = [Neuron(random()) for _ in range(input_count)]
        hidden_layer = [Neuron(random()) for _ in range(hidden_count)]
        out_layer = [Neuron(random()) for _ in range(out_count)]

        self.network = [input_layer, hidden_layer, out_layer]

    def activate(self):
        for i in range(1, len(self.network)):
            prev_layer = self.network[i - 1]
            current_layer = self.network



    def __str__(self):
        return '{}\n{}\n{}\n'.format(*self.network)


network = NeuralNetwork(2, 1, 2)
print(network)
