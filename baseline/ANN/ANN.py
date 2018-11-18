import numpy as np
from sklearn.model_selection import train_test_split
from Layer import Layer


def sigmoid(x):
    return np.divide(1, np.add(1, np.exp(np.multiply(-1, x))))


def tanh(x):
    return np.tanh(x)


def softmax(x):
    exp_x = np.exp(x)
    return np.divide(exp_x, np.sum(exp_x))


class ANN:
    def __init__(self, hidden_architecture, output_activation,
                 fitness_function, data, random_state,
                 validation_p=0.2, target_labels=None):
        self.hidden_architecture = hidden_architecture
        self.n_input_neurons = data[0].shape[1]
        self.fitness_function = fitness_function

        if target_labels is None:
            # regression
            self.n_output_neurons = 1
            self.output_activation = None
        elif len(target_labels) == 2:
            # binary classification
            self.n_output_neurons = 1
            self.output_activation = output_activation
            self.target_labels = target_labels
        else:
            # multi-label classification
            self.n_output_neurons = len(target_labels)
            self.output_activation = output_activation
            self.target_labels = target_labels

        if 0 < validation_p < 1:  # leave p% out cross-validation
            self.X_train, self.X_validate, self.y_train, self.y_validate = train_test_split(data[0], data[1],
                test_size=validation_p, random_state=random_state)
        else:  # no cross-validation
            self.X_train, self.X_validate = data[0], None
            self.y_train, self.y_validate = data[1], None

        self._initialize()

    def _initialize(self):
        # at least one hidden layer
        if not np.any((self.hidden_architecture[:, 0] <= 0)):
            # input layer
            self.network = [Layer(self.hidden_architecture[0, 0],
                                  self.n_input_neurons,
                                  self.hidden_architecture[0, 1])]
            # hidden layers
            self.network += [Layer(self.hidden_architecture[l, 0],
                                   self.hidden_architecture[l - 1, 0],
                                   self.hidden_architecture[l, 1])
                             for l in range(1, len(self.hidden_architecture))]
            # output layer
            if self.n_output_neurons > 1:
                # multi-label classification
                self.network += [Layer(self.n_output_neurons,
                                       self.hidden_architecture[len(self.hidden_architecture) - 1, 0],
                                       None)]
            else:
                # binary classification or regression
                self.network += [Layer(self.n_output_neurons,
                                       self.hidden_architecture[len(self.hidden_architecture) - 1, 0],
                                       self.output_activation)]

                # no hidden layer
        else:
            if self.n_output_neurons > 1:
                # multi-label classification
                self.network = [Layer(self.n_output_neurons, self.n_input_neurons, None)]
            else:
                # binary classification or regression
                self.network = [Layer(self.n_output_neurons, self.n_input_neurons, self.output_activation)]

    def stimulate(self, weights):
        self._set_weights(weights)

        fitness = []

        for signals, target in zip((self.X_train, self.X_validate), (self.y_train, self.y_validate)):
            if signals is None or signals is None:
                fitness.append(None)
                break

            for layer in self.network:
                signals = np.transpose(np.vstack(layer.activate(signals)))

            if self.n_output_neurons > 1:
                # multi-class classification
                signals = np.array([self.output_activation(i) for i in signals])
                signals = np.argmax(signals, axis=len(signals.shape) - 1)
                y_pred = self.target_labels[signals]
            elif self.output_activation is not None:
                # binary classification
                y_pred = [self.target_labels[0] if i < 0.5 else self.target_labels[1] for i in signals]
            else:
                # regression
                y_pred = signals

            fitness.append(self.fitness_function(target, y_pred))

        return fitness

    def stimulate_with(self, X, probabilities=True):
        signals = X

        for layer in self.network:
            signals = np.transpose(np.vstack(layer.activate(signals)))

        if self.n_output_neurons > 1:
            # multi-class classification
            signals = np.array([self.output_activation(i) for i in signals])
            signals = np.argmax(signals, axis=len(signals.shape) - 1)
            if probabilities:
                return signals
            else:
                return self.target_labels[signals]
        else:
            return signals

    def _set_weights(self, weights):
        i = 0
        for layer in self.network:
            n = layer.n_neurons * layer.n_dendrites
            layer.set_weights(weights[i:i + n])
            i+=n