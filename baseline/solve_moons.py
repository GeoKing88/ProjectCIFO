import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

import utils as uls
from problems.ANNOP import ANNOP
from ANN.ANN import ANN, sigmoid, tanh
from algorithms.genetic_algorithm import GeneticAlgorithm


# setup random state
seed = 1
random_state = uls.get_random_state(seed)

#++++++++++++++++++++++++++
# THE DATA
#++++++++++++++++++++++++++
# import data
X, y = datasets.make_moons(500, noise=.10, random_state=1)
#X, y = datasets.make_circles(n_samples=500, factor=.5, noise=.05)

print(X.shape)
plt.scatter(X[:, 0], X[:, 1], s=100, c=y, cmap=plt.cm.Spectral, edgecolor="black", linewidths=1)
plt.title("Scatterplot of red balls and blue balls")
plt.show()

#++++++++++++++++++++++++++
# THE ANN
#++++++++++++++++++++++++++
# ann's ingridients
from sklearn.metrics import accuracy_score
hl1 = 4
hl2 = 4
hl3 = 4
hl4 = 4
hidden_architecture = np.array([[hl1, sigmoid], [hl2, sigmoid], [hl3, sigmoid], [hl4, sigmoid]])
n_weights = X.shape[1]*hl1*hl2*hl3*hl4*len(set(y))
# create ann
ann_i = ANN(hidden_architecture, sigmoid, accuracy_score, (X, y), random_state, 0, (0, 1))

#++++++++++++++++++++++++++
# THE PROBLEM INSTANCE
#++++++++++++++++++++++++++
ann_op_i = ANNOP(search_space=(-2, 2, n_weights), fitness_function=uls.parametrized_ann(ann_i), minimization=False)

#++++++++++++++++++++++++++
# THE OPTIMIZATION
#++++++++++++++++++++++++++
n_gen = 50
ps = 100
p_c = 0.5
p_m = 0.9
radius = 0.2
ga = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(0.1),
                      uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(radius), p_c)
ga.initialize()
ga.search(n_gen, True)

ga.best_solution.print_()
print("Training fitness of the best solution: %.2f" % ga.best_solution.fitness)

#++++++++++++++++++++++++++
# VISUALIZE FIT
#++++++++++++++++++++++++++
x1_start, x1_stop = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_start, x2_stop = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

step = 0.01

x1_dim = np.arange(x1_start, x1_stop, step)
x2_dim = np.arange(x2_start, x2_stop, step)

xx, yy = np.meshgrid(x1_dim, x2_dim)
grid = np.c_[xx.ravel(), yy.ravel()]

ann_i._set_weights(ga.best_solution.representation)
Z = ann_i.stimulate_with(grid)
Z = np.where(Z > .5, 1, 0)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, s=100, edgecolor="black", linewidths=1)
plt.show()