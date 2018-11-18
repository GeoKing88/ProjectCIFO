import numpy as np
from continuous import Continuous
from solutions.solution import Solution


class ANNOP(Continuous):
    def __init__(self, search_space, fitness_function, minimization=True, validation_threshold=None):
        Continuous.__init__(self, search_space, fitness_function, minimization)
        self.validation_threshold = validation_threshold

    def evaluate(self, solution):
        solution.fitness, solution.validation_fitness = self.fitness_function(solution.representation)
        solution.dimensionality, solution.valid = self._validate(solution)

        if not solution.valid:
            if self.minimization:
                solution.fitness = np.iinfo(np.int32).max
            else:
                solution.fitness = 0

    def _validate(self, solution):
        if solution.validation_fitness is not None and self.validation_threshold is not None:
            return len(solution.representation), \
                   True if np.absolute(solution.fitness - solution.validation_fitness) <= self.validation_threshold else False
        else:
            return len(solution.representation), True

    def sample_search_space(self, random_state):
        return Solution(random_state.uniform(low=self.search_space[0], high=self.search_space[1],
                                             size=self.search_space[2]))