import logging
import numpy as np

from algorithms.hill_climbing import HillClimbing


class SimulatedAnnealing(HillClimbing):
    def __init__(self, problem_instance, random_state,
                 neighborhood_size, neighborhood_function,
                 control, update_rate):
        HillClimbing.__init__(self, problem_instance, random_state,
                              neighborhood_size, neighborhood_function)
        self.control = control
        self.update_rate = update_rate

    def search(self, n_iterations, report=False, log=False):
        if log:
            log_event = [self.problem_instance.__class__, id(self._random_state), __name__]
            logger = logging.getLogger(','.join(list(map(str, log_event))))

        i = self.best_solution

        for iteration in range(n_iterations):
            for neighbor in range(self.neighborhood_size):
                j = self._choose_random_neighbor(i)
                i = self._get_best(i, j)

            self._update_control_parameter()

            if report:
                self._verbose_reporter_inner(i, iteration)

            if log:
                log_event = [iteration, i.fitness, i.validation_fitness if hasattr(i, 'validation_fitness') else None,
                             self.neighborhood_size, self.neighborhood_function.__name__,
                             self.control, self.update_rate]
                logger.info(','.join(list(map(str, log_event))))

        self.best_solution = i

    def _get_best(self, candidate_a, candidate_b):
        if self.problem_instance.minimization:
            if candidate_a.fitness >= candidate_b.fitness:
                return candidate_b
            elif candidate_b.valid and np.exp(
                    (candidate_a.fitness - candidate_b.fitness) / self.control) > self._random_state.uniform():
                return candidate_b
            else:
                return candidate_a
        else:
            if candidate_a.fitness <= candidate_b.fitness:
                #print("Accepted best: %.2f" % (candidate_b.fitness))
                return candidate_b
            elif candidate_b.valid and np.exp(
                    -(candidate_a.fitness - candidate_b.fitness) / self.control) > self._random_state.uniform():
                #print("Accepted worse: %.2f against %.2f with p=%.3f" % (candidate_a.fitness, candidate_b.fitness, np.exp(-(candidate_a.fitness-candidate_b.fitness)/self.control)))
                return candidate_b
            else:
                #print("Invalid")
                return candidate_a

    def _update_control_parameter(self):
        self.control *= self.update_rate