class SearchAlgorithm:
    def __init__(self, problem_instance):
        self.problem_instance = problem_instance


    def initialize(self):
        pass


    def search(self, n_iterations, report=False):
        pass


    def _get_best(self, candidate_a, candidate_b):
        if self.problem_instance.minimization:
            if candidate_a.fitness > candidate_b.fitness:
                return candidate_b
            else:
                return candidate_a
        else:
            if candidate_a.fitness < candidate_b.fitness:
                return candidate_b
            else:
                return candidate_a


    def verbose_reporter(self):
        print("Best solution found:")
        self.best_solution.print_()


    def _verbose_reporter_inner(self, solution, iteration):
        print("> > > Current best solution at iteration %d:" % iteration)
        solution.print_()
        print()