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
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.genetic_algorithm2 import GeneticAlgorithm2
from hill_climbing import HillClimbing
from simulated_annealing import SimulatedAnnealing
from algorithms.genetic_algorithm3 import GeneticAlgorithm3
import math as math
import pandas as pd
import openpyxl
from openpyxl.utils import get_column_letter

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
seed =78
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
ps = 50
p_c = .4
p_m = 0.9
radius = .2
pressure = .2
p_migration = 0
ga1 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrize_roulette_wheel(pressure),
                          uls.geometric_semantic_crossover, p_c, uls.parametrized_ball_mutation(radius), p_m, pressure)

ga2 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrize_botzmann_selection(pressure),
                          uls.geometric_semantic_crossover, p_c, uls.parametrized_ball_mutation(radius), p_m, pressure)

ga3 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrize_botzmann_selection(0.2),
                          uls.geometric_semantic_crossover, p_c, uls.parametrized_ball_mutation(radius), p_m, 0.2)

ga4 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrize_botzmann_selection(0.2),
                          uls.geometric_semantic_crossover, p_c, uls.parametrized_ball_mutation(radius), p_m, 0.2)

ga5 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrize_botzmann_selection(0.2),
                          uls.geometric_semantic_crossover, p_c, uls.parametrized_ball_mutation(radius), p_m, 0.2)

'''
ga2 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure=pressure),
                       uls.two_point_crossover, p_c, uls.parametrized_ball_mutation(radius), p_m)
'''
islands =[]
ga1.initialize()
islands.append(ga1)
#ga2.initialize()
#islands.append(ga2)
best_solution = ga1.best_solution
#print(set((np.around(best_solution.representation,2)).tolist()))
#print(ga2.best_solution.fitness)
'''
for iteration in range(20):
    print("AQUI")
    ga1.search(40, False, False)
    ga2.search(40, False, False)

    for i in range(len(islands)):
        if best_solution.fitness < islands[i].best_solution.fitness:
            algorithm = islands[i]
            best_solution = islands[i].best_solution

    print(">>>>>>>>>>>>>>>>>INTERATION: ",iteration)
    print(">>>>>>>>>>>>>>>>>BEST_SOLUTION: ",best_solution.id)
    print("G11: ", ga1.best_solution1.fitness)
    print("G12: ", ga1.best_solution2.fitness)
    print("G21: ", ga2.best_solution1.fitness)
    print("G22: ", ga2.best_solution2.fitness)
    print(">>>>>>>>>>>>>>>>>FITNESS>>>>>>>>>>>>>>>>>>>>: ",str(round(best_solution.fitness, 2)))
best_solution.print_()

gt = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrize_roulette_wheel(0.6),
                          uls.two_point_crossover, p_c, uls.parametrized_ball_mutation(radius), p_m)

gt.population = ga1.population + ga2.population


for iteration in range(100):

    print("AQUI2")
    gt.best_solution = best_solution
    gt.search(20, True, False)
    if gt.best_solution.fitness > best_solution.fitness:
        best_solution = gt.best_solution
    print(">>>>>>>>>>>>>>>>>INTERATION: ", iteration)
    print(">>>>>>>>>>>>>>>>>BEST_SOLUTION: ", best_solution.id)
    print("G1: ", gt.best_solution.fitness)
    print(">>>>>>>>>>>>>>>>>FITNESS>>>>>>>>>>>>>>>>>>>>: ", str(round(best_solution.fitness, 2)))


random_state.shuffle(gt.population)
ga1.population1 = gt.population[:13]
ga1.population2 = gt.population[14:25]
ga2.population1 = gt.population[25:38]
ga2.population2 = gt.population[38:50]
'''
constat = 2
#wb = openpyxl.Workbook()
#ws1 = wb.create_sheet("Mysheet")
#ws2 = wb.create_sheet("2")
save_solutions = []
#save_solutions2 = []
save_solutions.append(ga1.best_solution)
#save_solutions2.append(ga2.best_solution)
for iteration in range(n_gen):
    ga1.search(100, False, False)
    #ga2.search(100, False, False)
    if iteration % 20 ==0 and iteration !=0:
        '''print("TROCA")

        ga1.population = uls.order_numpy_solutions_array(ga1.population)
        ga2.population = uls.order_numpy_solutions_array(ga2.population)

        solution1 = ga1.population[:3].copy()
        solution2 = ga2.population[:3].copy()

        ga1.population[:3] = solution2
        ga2.population[:3] = solution1
        '''
        '''
        ga1.population = uls.order_numpy_solutions_array(ga1.population)
        ga2.population = uls.order_numpy_solutions_array(ga2.population)
        ga3.population = uls.order_numpy_solutions_array(ga3.population)
        ga4.population = uls.order_numpy_solutions_array(ga4.population)
        ga5.population = uls.order_numpy_solutions_array(ga5.population)

        solution1 = ga1.population[:4].copy()
        solution2 = ga2.population[:4].copy()
        solution2 = ga2.population[:4].copy()
        solution2 = ga2.population[:4].copy()
        solution2 = ga2.population[:4].copy()

        ga1.population[:4] = solution2
        ga2.population[:4] = solution1
        '''
    for i in range(len(islands)):
        if best_solution.fitness < islands[i].best_solution.fitness:
            algorithm = islands[i]
            best_solution = islands[i].best_solution

    print(">>>>>>>>>>>>>>>>>INTERATION: ",iteration)
    print(">>>>>>>>>>>>>>>>>BEST_SOLUTION: ",best_solution.id)
    print("\n")
    print("G1: ", ga1.best_solution.fitness)
    print("G1 MEAN: ", np.mean(ga1.get_all_fitness()))
    #print("G1 STD: ", np.std(ga1.get_all_fitness()))
    #print("\n")
    #print("G2: ", ga2.best_solution.fitness)

    #print("G2 MEAN: ", np.mean(ga2.get_all_fitness()))
    #print("G2 STD: ", np.std(ga2.get_all_fitness()))
    print(">>>>>>>>>>>>>>>>>FITNESS>>>>>>>>>>>>>>>>>>>>: ", str(round(best_solution.fitness, 2)))
    #save_solutions.append(ga1.best_solution)
    #save_solutions2.append(ga2.best_solution)
    row = 1
    column = 0
    '''
    for i in range(len(ga1.best_solution.representation)):
        ws1[get_column_letter(iteration+1) + str(row)] = ga1.best_solution.representation[i]
        ws2[get_column_letter(iteration+1) + str(row)] = ga2.best_solution.representation[i]
        row += 1
    ws1[get_column_letter(iteration+4) + str(row+1)] = ga1.best_solution.fitness
    ws2[get_column_letter(iteration + 4) + str(row + 1)] = ga2.best_solution.fitness
    

wb.save('C:\\Users\\hppor\\Desktop\\Template.xlsx')
'''
best_solution.print_()
'''
gt.presure1 = 0.9
#gt.population = ga1.population1 + ga2.population2
for iteration in range(10):
    print("AQUI4")
    gt.best_solution = best_solution
    gt.search(20, True, False)
    if gt.best_solution.fitness > best_solution.fitness:
        best_solution = gt.best_solution
    print(">>>>>>>>>>>>>>>>>INTERATION: ",iteration)
    print(">>>>>>>>>>>>>>>>>BEST_SOLUTION: ", best_solution.id)
    print("G1: ", gt.best_solution.fitness)
    print(">>>>>>>>>>>>>>>>>FITNESS>>>>>>>>>>>>>>>>>>>>: ", str(round(best_solution.fitness, 2)))
'''
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


'''
ga = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                          uls.subtoru_exchange_crossover, p_c, uls.parametrized_ball_mutation(radius), p_m)


ga2 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure=pressure),
                       uls.two_point_crossover, p_c, uls.parametrized_ball_mutation(radius), p_m)

ga.initialize()
ga.search(n_gen, True, True)
teste1 = ga.get_best_solution()



ga.best_solution.print_()


print("Training fitness of the best solution: %.2f" % ga.best_solution.fitness)
print("Validation fitness of the best solution: %.2f" % ga.best_solution.validation_fitness)


#++++++++++++++++++++++++++
# TEST
#++++++++++++++++++++++++++
ann_i._set_weights(ga.best_solution.representation)
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

ann_i._set_weights(ga.best_solution.representation)
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

import numpy as np
import matplotlib.pyplot as plt
x = np.asarray(ga.list_iteration)
y = np.asarray([x*100 for x in ga.get_best_solution()])
plt.scatter(x, y)
plt.xticks(x)
plt.show()


ga = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                          uls.uniform_points_crossover, p_c, uls.parametrized_ball_mutation(radius), p_m)



ga.initialize()
ga.search(n_gen, True, True)
teste2 = ga.get_best_solution()



ga.best_solution.print_()


print("Training fitness of the best solution: %.2f" % ga.best_solution.fitness)
print("Validation fitness of the best solution: %.2f" % ga.best_solution.validation_fitness)


#++++++++++++++++++++++++++
# TEST
#++++++++++++++++++++++++++
ann_i._set_weights(ga.best_solution.representation)
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

ann_i._set_weights(ga.best_solution.representation)
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


import numpy as np
import matplotlib.pyplot as plt
x = np.asarray(ga.list_iteration)
y = np.asarray([x*100 for x in ga.get_best_solution()])
plt.scatter(x, y)
plt.xticks(x)
plt.show()
'''
