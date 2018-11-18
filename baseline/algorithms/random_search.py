import logging

from search_algorithm import SearchAlgorithm


class RandomSearch(SearchAlgorithm):
    def __init__(self, problem_instance, random_state):
        SearchAlgorithm.__init__(self, problem_instance)
        self._random_state = random_state

    def initialize(self):
        self.best_solution = self._generate_random_valid_solution()

    def search(self, n_iterations, report=False, log=False):
        if log:
            log_event = [self.problem_instance.__class__,  id(self._random_state), __name__]
            logger = logging.getLogger(','.join(list(map(str, log_event))))

        i = self.best_solution
        for iteration in range(n_iterations):
            j = self._generate_random_solution()
            i = self._get_best(i, j)

            if report:
                self._verbose_reporter_inner(i, iteration)

            if log:
                log_event = [iteration, i.fitness, i.validation_fitness if hasattr(i, 'validation_fitness') else None]
                logger.info(','.join(list(map(str, log_event))))

        self.best_solution = i

    def _generate_random_solution(self):
        return self.problem_instance.sample_search_space(self._random_state)

    def _generate_random_valid_solution(self):
        solution = self._generate_random_solution()
        self.problem_instance.evaluate(solution)

        while not solution.valid:
            solution = self._generate_random_solution()
            self.problem_instance.evaluate(solution)

        return solution
