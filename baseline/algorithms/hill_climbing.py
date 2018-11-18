import logging

from utils import bit_flip
from random_search import RandomSearch
from solutions.solution import Solution


class HillClimbing(RandomSearch):
    def __init__(self, problem_instance, random_state,
                 neighborhood_size, neighborhood_function=bit_flip):
        RandomSearch.__init__(self, problem_instance, random_state)
        self.neighborhood_size = neighborhood_size
        self.neighborhood_function = neighborhood_function

    def search(self, n_iterations, report=False, log=False):
        if log:
            log_event = [self.problem_instance.__class__, id(self._random_state), __name__]
            logger = logging.getLogger(','.join(list(map(str, log_event))))

        i = self.best_solution
        for iteration in range(n_iterations):
            j = self._explore_neighborhood(i)
            i = self._get_best(i, j)

            if report:
                self._verbose_reporter_inner(i, iteration)

            if log:
                log_event = [iteration, i.fitness, i.validation_fitness if hasattr(i, 'validation_fitness') else None,
                             self.neighborhood_size, self.neighborhood_function.__name__]
                logger.info(','.join(list(map(str, log_event))))

        self.best_solution = i

    def _explore_neighborhood(self, solution):
        best_neighbor = self._choose_random_neighbor(solution)
        for neighbor in range(self.neighborhood_size - 1):
            candidate_neighbor = self._choose_random_neighbor(solution)
            best_neighbor = self._get_best(best_neighbor, candidate_neighbor)
        return best_neighbor

    def _choose_random_neighbor(self, solution):
        neighbor = Solution(self.neighborhood_function(solution.representation,
                                                       self._random_state))
        self.problem_instance.evaluate(neighbor)
        return neighbor
