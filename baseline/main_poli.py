import os
import datetime

import logging
import numpy as np

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import utils as uls
from problems.ANNOP import ANNOP
from ANN.ANN import ANN, softmax, sigmoid
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.genetic_algorithm2 import GeneticAlgorithm2
from algorithms.simulated_annealing import SimulatedAnnealing


# setup logger
file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "LogFiles/" + (str(datetime.datetime.now().date()) + "-" + str(datetime.datetime.now().hour) + \
            "_" + str(datetime.datetime.now().minute) + "_log.csv"))
logging.basicConfig(filename=file_path, level=logging.DEBUG, format='%(name)s,%(message)s')

# ++++++++++++++++++++++++++
# THE DATA
# restrictions:
# - MNIST digits (8*8)
# - 33% for testing
# - flattened input images
# ++++++++++++++++++++++++++
digits = datasets.load_digits()
flat_images = np.array([image.flatten() for image in digits.images])

# split data
X_train, X_test, y_train, y_test = train_test_split(flat_images, digits.target, test_size=0.33, random_state=0)

# setup benchmarks
n_runs = 2
n_gen = 100
validation_p = .2
validation_threshold = .07

# Genetic Algorithm setup
ps = 50
p_c = 0.9
radius = .2
pressure = 0.2

# Simulated Annealing setup
ns = ps
control = 2
update_rate = 0.9

for seed in range(n_runs):
    random_state = uls.get_random_state(seed)

    #++++++++++++++++++++++++++
    # THE ANN
    # restrictions:
    # - 2 h.l. with Sigmoid a.f.
    # - Softmax a.f. at output
    # - 20%, out of remaining 67%, for validation
    #++++++++++++++++++++++++++
    # ann's architecture
    hidden_architecture = np.array([[10, sigmoid], [10, sigmoid]])
    n_weights = X_train.shape[1]*10*10*len(digits.target_names)
    # create ann
    ann_i = ANN(hidden_architecture, softmax, accuracy_score, (X_train, y_train), random_state, validation_p, digits.target_names)

    #++++++++++++++++++++++++++
    # THE PROBLEM INSTANCE
    # - optimization of ANN's weights is a COP
    #++++++++++++++++++++++++++
    ann_op_i = ANNOP(search_space=(-2, 2, n_weights), fitness_function=ann_i.stimulate,
                     minimization=False, validation_threshold=validation_threshold)

    #++++++++++++++++++++++++++
    # THE SEARCH
    # restrictions:
    # - 5000 offsprings/run max*
    # - 50 offsprings/generation max*
    # - use at least 5 runs for your benchmarks
    # * including reproduction
    #++++++++++++++++++++++++++
    ga1 = GeneticAlgorithm2(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                          uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(radius), 0.2)
    ga2 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                           uls.two_point_crossover, p_c, uls.parametrized_ball_mutation(radius), 0.1)
    sa1 = SimulatedAnnealing(ann_op_i, random_state, ps, uls.parametrized_ball_mutation(radius), control, update_rate)
    search_algorithms = [ga1, ga2]

    # initialize search algorithms
    [algorithm.initialize() for algorithm in search_algorithms]

    # execute search
    [algorithm.search(n_iterations=n_gen, report=False) for algorithm in search_algorithms]

#++++++++++++++++++++++++++
# TEST
# - test algorithms on unseen data
#++++++++++++++++++++++++++
for algorithm in search_algorithms:
    ann_i._set_weights(algorithm.best_solution.representation)
    y_pred = ann_i.stimulate_with(X_test, False)
    accuracy = accuracy_score(y_test, y_pred)
    print("Unseen Accuracy of %s: %.2f" % (algorithm.__class__, accuracy_score(y_test, y_pred)))