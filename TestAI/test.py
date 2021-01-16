import numpy as np
import matplotlib.pyplot as plt
import random

def activate(x):
    if x >= 0:
        return 1
    else:
        return 0


inp = [[random.randint(-10, 10), random.randint(-10, 10)] for i in range(300)]

out_neuron = [-0.3, 0.3, -0.3 * 2]

for i in inp:
    out = np.dot(out_neuron, i)

    if activate(out):
        plt.scatter(*i, color='green', s=1)
    else:
        plt.scatter(*i, color='blue', s=1)

plt.show()