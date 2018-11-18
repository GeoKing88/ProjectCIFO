import numpy as np


class Neuron:
    _id = 0

    def __init__(self, n_dendrites, activation):
        self._neuron_id = Neuron._id
        Neuron._id += 1
        self.n_dendrites = n_dendrites
        self.weights = None  # np.zeros(self.n_dendrites)
        self.activation = activation

    def __str__(self):
        return "Neuron#%d weights: %s" % (self._neuron_id, str(self.weights))

    def activate(self, signals):
        weighted_signals = np.multiply(self.weights, signals)
        sum_weighted_signals = np.sum(weighted_signals, axis=len(weighted_signals.shape) - 1)
        if self.activation is None:
            return sum_weighted_signals
        else:
            return self.activation(sum_weighted_signals)
