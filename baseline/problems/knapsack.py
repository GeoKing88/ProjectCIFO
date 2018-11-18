import numpy as np
from problem import Problem
from solutions._item import sum_weights
from solutions.solution import Solution
from utils import random_boolean_1D_array


class Knapsack(Problem):
    def __init__(self, capacity, search_space, fitness_function, minimization=False):
        Problem.__init__(self, search_space, fitness_function, minimization)
        self.capacity = capacity


    def evaluate(self, solution):
        items = self.search_space[solution.representation]
        solution.weight, solution.dimensionality, solution.valid = self._validate(items)

        if solution.valid:
            solution.fitness = self.fitness_function(items)
        else:
            if self.minimization:
                solution.fitness = np.iinfo(np.int32).max
            else:
                solution.fitness = 0


    def _validate(self, items):
        weight = sum_weights(items)
        dimensionality = len(items)
        if weight <= self.capacity:
            return weight, dimensionality, True
        else:
            return weight, dimensionality, False


    def sample_search_space(self, random_state):
        return Solution(random_boolean_1D_array(self.dimensionality, random_state))