from Neuron import Neuron

class Layer:
    _id = 0

    def __init__(self, n_neurons, n_dendrites, activation):
        self._layer_id = Layer._id
        Layer._id += 1
        self.n_neurons = n_neurons
        self.n_dendrites = n_dendrites
        self.activation = activation
        self.neurons = [Neuron(self.n_dendrites, self.activation)
                        for _ in range(0, self.n_neurons)]

    def __str__(self):
        return 'Layer:\t\t' + '\n\t\t'.join([str(neuron) for neuron in self.neurons])

    def activate(self, signals):  # [[outputs of N0], [outputs of N1], ...]
        return [neuron.activate(signals) for neuron in self.neurons]

    def get_weights(self):
        return [neuron.weights for neuron in self.neurons]

    def set_weights(self, weights):
        i = 0
        for neuron in self.neurons:
            neuron.weights = weights[i:i + self.n_dendrites]
            i+=self.n_dendrites