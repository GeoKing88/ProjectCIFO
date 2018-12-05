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
from algorithms.genetic_algorithm4 import GeneticAlgorithm

# ++++++++++++++++++++++++++
# THE DATA
# restrictions:
# - MNIST digits (8*8)
# - 33% for testing
# - flattened input images
# ++++++++++++++++++++++++++
# import data
digits = datasets.load_digits()
flat_images = np.array([image.flatten() for image in digits.images])

#print(flat_images.shape)
#print(digits.target_names)

make_plots = True

if make_plots:
    n_images = 25
    plt.figure(figsize=(10, 10))
    for i in range(n_images):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(digits.images[i], cmap=plt.cm.binary)
        plt.xlabel("Value: %d" % digits.target_names[digits.target[i]], fontsize=12)
    plt.suptitle('Example of the training data', fontsize=30)
    plt.show()

# setup random state
seed = 666
random_state = uls.get_random_state(seed)

# split data
X_train, X_test, y_train, y_test = train_test_split(flat_images, digits.target, test_size=0.33,
                                                    random_state=random_state)

# ++++++++++++++++++++++++++
# THE ANN
# restrictions:
# - 2 h.l. with sigmoid a.f.
# - softmax a.f. at output
# - 20% for validation
# ++++++++++++++++++++++++++
# ann's ingridients
hl1 = 10
hl2 = 10
hidden_architecture = np.array([[hl1, sigmoid], [hl2, sigmoid]])
n_weights = X_train.shape[1] * hl1 * hl2 * len(digits.target_names)
validation_p = 0.2
# create ann
ann_i = ANN(hidden_architecture, softmax, accuracy_score,
            (X_train, y_train), random_state, validation_p, digits.target_names)

# ++++++++++++++++++++++++++
# THE PROBLEM INSTANCE
# ++++++++++++++++++++++++++
validation_threshold = 0.07
ann_op_i = ANNOP(search_space=(-2, 2, n_weights), fitness_function=ann_i.stimulate,
                 minimization=False, validation_threshold=validation_threshold)

# ++++++++++++++++++++++++++
# THE OPTIMIZATION
# restrictions:
# - 5000 f.e./run
# - 50 f.e./generation
# - use at least 5 runs for benchmarks
# ++++++++++++++++++++++++++
n_gen = 100
ps = 50
p_c = .9
p_m = 0.4
radius = .9
pressure = .2
p_migration = 0
ga1 = GeneticAlgorithm(problem_instance=ann_op_i, random_state=random_state,
                       population_size=ps, selection=uls.parametrized_tournament_selection(0.2),
                       crossover=uls.geometric_semantic_crossover, p_c=p_c,
                       mutation=uls.parametrized_ball_mutation(radius), p_m=p_m, pressure=pressure)

ga2 = GeneticAlgorithm(problem_instance=ann_op_i, random_state=random_state,
                       population_size=ps, selection=uls.parametrize_roulette_wheel_w_pressure(0.2),
                       crossover=uls.geometric_semantic_crossover, p_c=p_c,
                       mutation=uls.parametrized_ball_mutation(radius), p_m=p_m, pressure=pressure)

ga3 = GeneticAlgorithm(problem_instance=ann_op_i, random_state=random_state,
                       population_size=ps, selection=uls.parametrize_roulette_wheel_w_pressure(0.2),
                       crossover=uls.geometric_semantic_crossover, p_c=p_c,
                       mutation=uls.parametrized_ball_mutation(radius), p_m=p_m, pressure=pressure)

ga4 = GeneticAlgorithm(problem_instance=ann_op_i, random_state=random_state,
                       population_size=ps, selection=uls.parametrize_roulette_wheel_w_pressure(0.2),
                       crossover=uls.geometric_semantic_crossover, p_c=p_c,
                       mutation=uls.parametrized_ball_mutation(radius), p_m=p_m, pressure=pressure)


islands = []
ga1.initialize()
islands.append(ga1)
best_solution = ga1.best_solution
'''ga2.initialize()
islands.append(ga2)
ga3.initialize()
islands.append(ga3)
ga4.initialize()
islands.append(ga4)


print(ga1.distance_between_populations(ga1.average_distance_population_normalize(), ga2.average_distance_population_normalize()))
print(ga1.distance_between_populations(ga1.average_distance_population_normalize(), ga3.average_distance_population_normalize()))
print(ga1.distance_between_populations(ga1.average_distance_population_normalize(), ga4.average_distance_population_normalize()))
print(ga2.distance_between_populations(ga2.average_distance_population_normalize(), ga3.average_distance_population_normalize()))
print(ga2.distance_between_populations(ga2.average_distance_population_normalize(), ga4.average_distance_population_normalize()))
print(ga3.distance_between_populations(ga3.average_distance_population_normalize(), ga4.average_distance_population_normalize()))
'''
for iteration in range(n_gen):
    print("AQUI")
    ga1.search(100, False, False)
    #ga2.search(100, False, False)
    #ga3.search(100, False, False)
    #ga4.search(100, False, False)

    '''
    if iteration == 20:
        ga1.flag = True
        #ga2.flag = True
        #ga3.flag = True
        #ga4.flag = True
    '''
    for i in range(len(islands)):
        if best_solution.fitness < islands[i].best_solution.fitness:
            algorithm = islands[i]
            best_solution = islands[i].best_solution

    print(">>>>>>>>>>>>>>>>>INTERATION: ", iteration)
    print(">>>>>>>>>>>>>>>>>BEST_SOLUTION: ", best_solution.id)
    print("G1: ", [ga1.best_solution.fitness, round(uls.calculate_media_solution(ga1.population), 2)], sep='\t')
    #print("G2: ", [ga2.best_solution.fitness, round(uls.calculate_media_solution(ga2.population), 2)], sep='\t')
    #print("G3: ", [ga3.best_solution.fitness, round(uls.calculate_media_solution(ga3.population), 2)], sep='\t')
    #print("G4: ", [ga4.best_solution.fitness, round(uls.calculate_media_solution(ga4.population), 2)], sep='\t')
    print(">>>>>>>>>>>>>>>>>FITNESS>>>>>>>>>>>>>>>>>>>>: ", str(round(best_solution.fitness, 2)))

best_solution.print_()

print("Training fitness of the best solution: %.2f" % best_solution.fitness)
print("Validation fitness of the best solution: %.2f" % best_solution.fitness)

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