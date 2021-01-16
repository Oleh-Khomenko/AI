import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

random.seed(1)


def sigmoid(x):
    return 1 / (1 + np.exp(x))


class Neuron:
    def __init__(self, weights, bias=0):
        self.weights = weights
        self.bias = bias

    def feed_forward(self, inp):
        out = np.dot(inp, self.weights) + self.bias
        return sigmoid(out)


class NeuralNetwork:
    def __init__(self, hidden_counts, out_count):
        self.hidden_layers = [[Neuron([random.random(), random.random()])
                               for _ in range(hidden_counts[j])] for j in range(len(hidden_counts))]
        self.out_neurons = [Neuron([random.random(), random.random()]) for _ in range(out_count)]
        self.lamb = 0.01

    def feed_forward(self, inp):
        inputs = inp
        for layer in self.hidden_layers:
            res = []
            for neuron in layer:
                res.append(neuron.feed_forward(inputs))
            inputs = res

        out = []
        for neuron in self.out_neurons:
            out.append(neuron.feed_forward(inputs))

        return out

    def back_propagation(self, out_neuron, out):
        """count of out neurons = 1"""

        e = out - out_neuron
        grad = 0

    def train(self, inp, out):
        pass
