import utils as uls
import numpy as np
from problem import Problem
from solutions.solution import Solution


class Continuous(Problem):
    def __init__(self, search_space, fitness_function, minimization=True):
        Problem.__init__(self, search_space, fitness_function, minimization)

    def evaluate(self, solution):
        point = solution.representation
        solution.dimensionality, solution.valid = self._validate(point)

        if solution.valid:
            solution.fitness = self.fitness_function(point)
        else:
            if self.minimization:
                solution.fitness = np.iinfo(np.int32).max
            else:
                solution.fitness = 0

    def _validate(self, point):
        dimensionality = len(point)

        for dim in range(dimensionality):
            if point[dim] < self.search_space[dim][0] or point[dim] > self.search_space[dim][1]:
                return dimensionality, False

        return dimensionality, True

    def sample_search_space(self, random_state):
        return Solution(uls.random_float_1D_array(self.search_space, random_state))
