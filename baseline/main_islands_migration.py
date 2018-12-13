import os
import datetime

import logging
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import utils as uls
from problems.ANNOP import ANNOP
from ANN.ANN import ANN, softmax, sigmoid
from algorithms.genetic_algorithm_Elitism import GeneticAlgorithm
from algorithms.genetic_algorithm2 import GeneticAlgorithm2
from hill_climbing import HillClimbing
from simulated_annealing import SimulatedAnnealing
from algorithms.genetic_algorithm3 import GeneticAlgorithm3
import math as math
import pandas as pd
import openpyxl
from openpyxl.utils import get_column_letter
import copy

# setup logger
file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "LogFiles/" + (str(datetime.datetime.now().date()) + "-" + str(datetime.datetime.now().hour) + \
            "_" + str(datetime.datetime.now().minute) + "_log.csv"))
logging.basicConfig(filename=file_path, level=logging.DEBUG, format='%(name)s,%(message)s')

#++++++++++++++++++++++++++
# THE DATA
# restrictions:
# - MNIST digits (8*8)
# - 33% for testing
# - flattened input images
#++++++++++++++++++++++++++
# import data
digits = datasets.load_digits()
flat_images = np.array([image.flatten() for image in digits.images])

print(flat_images.shape)
print(digits.target_names)

make_plots = True

if make_plots:
    n_images = 25
    plt.figure(figsize=(10, 10))
    for i in range(n_images):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(digits.images[i], cmap=plt.cm.binary)
        plt.xlabel("Value: %d" % digits.target_names[digits.target[i]], fontsize=12)
    plt.suptitle('Example of the training data',  fontsize=30)
    plt.show()

# setup random state
seed =666
random_state = uls.get_random_state(seed)

# split data
X_train, X_test, y_train, y_test = train_test_split(flat_images, digits.target, test_size=0.33, random_state=random_state)

#++++++++++++++++++++++++++
# THE ANN
# restrictions:
# - 2 h.l. with sigmoid a.f.
# - softmax a.f. at output
# - 20% for validation
#++++++++++++++++++++++++++
# ann's ingridients
hl1 = 10
hl2 = 10
hidden_architecture = np.array([[hl1, sigmoid], [hl2, sigmoid]])
n_weights = X_train.shape[1]*hl1*hl2*len(digits.target_names)
validation_p = 0.2
# create ann
ann_i = ANN(hidden_architecture, softmax, accuracy_score,
                   (X_train, y_train), random_state, validation_p, digits.target_names)




#++++++++++++++++++++++++++
# THE PROBLEM INSTANCE
#++++++++++++++++++++++++++
validation_threshold = 0.07
ann_op_i = ANNOP(search_space=(-2, 2, n_weights), fitness_function=ann_i.stimulate,
                 minimization=False, validation_threshold=validation_threshold)

#++++++++++++++++++++++++++
# THE OPTIMIZATION
# restrictions:
# - 5000 f.e./run
# - 50 f.e./generation
# - use at least 5 runs for benchmarks
#++++++++++++++++++++++++++
n_gen = 100
ps = 10
p_c = .8
p_m = .8
radius = .002
pressure = .5

ga1 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                          uls.cicle_crossover, p_c, uls.parametrized_gaussian_member_mutation(radius)
                                                                                        , p_m, pressure)

ga2 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                          uls.cicle_crossover, p_c, uls.parametrized_gaussian_member_mutation(radius)
                                                                                        , p_m, pressure)

ga3 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                          uls.cicle_crossover, p_c, uls.parametrized_gaussian_member_mutation(radius)
                                                                                        , p_m, pressure)

ga4 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                          uls.cicle_crossover, p_c, uls.parametrized_gaussian_member_mutation(radius)
                                                                                        , p_m, pressure)

ga5 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                          uls.cicle_crossover, p_c, uls.parametrized_gaussian_member_mutation(radius)
                                                                                        , p_m, pressure)

islands =[]
ga1.initialize()
islands.append(ga1)
ga2.initialize()
islands.append(ga2)
ga3.initialize()
islands.append(ga3)
ga4.initialize()
islands.append(ga4)
ga5.initialize()
islands.append(ga5)
best_solution = ga1.best_solution


for iteration in range(100):
    ga1.search(40, False, False)
    ga2.search(40, False, False)

    for i in range(len(islands)):
        if best_solution.fitness < islands[i].best_solution.fitness:
            algorithm = islands[i]
            best_solution = islands[i].best_solution

        if iteration % 20 ==0 and iteration != 0:
            migration1 = copy.deepcopy(ga1.population[:3])
            migration2 = copy.deepcopy(ga2.population[:3])
            migration3 = copy.deepcopy(ga3.population[:3])
            migration4 = copy.deepcopy(ga4.population[:3])
            migration5 = copy.deepcopy(ga5.population[:3])

            ga1.population[:3] = migration2
            ga2.population[:3] = migration3
            ga3.population[:3] = migration4
            ga4.population[:3] = migration5
            ga5.population[:3] = migration1


    print('\n')
    print(">>>>>>>>>>>>>>>>>INTERATION: ", iteration)
    print(">>>>>>>>>>>>>>>>>FITNESS>>>>>>>>>>>>>>>>>>>>: ", str(ga1.best_solution.fitness))
    print(">>>>>>>>>>>>>>>>>G1 - MEAN: ", uls.calculate_media_solution(ga1.population))
    print(">>>>>>>>>>>>>>>>>FITNESS>>>>>>>>>>>>>>>>>>>>: ", str(ga2.best_solution.fitness))
    print(">>>>>>>>>>>>>>>>>G2 - MEAN: ", uls.calculate_media_solution(ga2.population))
    print(">>>>>>>>>>>>>>>>>FITNESS>>>>>>>>>>>>>>>>>>>>: ", str(ga3.best_solution.fitness))
    print(">>>>>>>>>>>>>>>>>G3 - MEAN: ", uls.calculate_media_solution(ga3.population))
    print(">>>>>>>>>>>>>>>>>FITNESS>>>>>>>>>>>>>>>>>>>>: ", str(ga4.best_solution.fitness))
    print(">>>>>>>>>>>>>>>>>G4 - MEAN: ", uls.calculate_media_solution(ga4.population))
    print(">>>>>>>>>>>>>>>>>FITNESS>>>>>>>>>>>>>>>>>>>>: ", str(ga5.best_solution.fitness))
    print(">>>>>>>>>>>>>>>>>G5 - MEAN: ", uls.calculate_media_solution(ga5.population))

print("Training fitness of the best solution: %.2f" % best_solution.fitness)
print("Validation fitness of the best solution: %.2f" % best_solution.validation_fitness)


#++++++++++++++++++++++++++
# TEST
#++++++++++++++++++++++++++
ann_i._set_weights(best_solution.representation)
y_pred = ann_i.stimulate_with(X_test, False)
print("Unseen Accuracy of the best solution: %.2f" % accuracy_score(y_test, y_pred))

if make_plots:
    n_images = 25
    images = X_test[0:n_images].reshape((n_images, 8, 8))
    f = plt.figure(figsize=(10, 10))
    for i in range(n_images):
        sub = f.add_subplot(5, 5, i + 1)
        sub.imshow(images[i], cmap=plt.get_cmap("Greens") if y_pred[i] == y_test[i] else plt.get_cmap("Reds"))
        plt.xticks([])
        plt.yticks([])
        sub.set_title('y^: %i, y: %i' % (y_pred[i], y_test[i]))
    f.suptitle('Testing classifier on unseen data')
    plt.show()

ann_i._set_weights(best_solution.representation)
y_pred = ann_i.stimulate_with(X_test, False)
print("Unseen Accuracy of the best solution: %.2f" % accuracy_score(y_test, y_pred))
if make_plots:
    n_images = 25
    images = X_test[0:n_images].reshape((n_images, 8, 8))
    f = plt.figure(figsize=(10, 10))
    for i in range(n_images):
        sub = f.add_subplot(5, 5, i + 1)
        sub.imshow(images[i], cmap=plt.get_cmap("Greens") if y_pred[i] == y_test[i] else plt.get_cmap("Reds"))
        plt.xticks([])
        plt.yticks([])
        sub.set_title('y^: %i, y: %i' % (y_pred[i], y_test[i]))
    f.suptitle('Testing classifier on unseen data')
    plt.show()