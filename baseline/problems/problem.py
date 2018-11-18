class Problem:
    def __init__(self, search_space, fitness_function, minimization):
        self.search_space = search_space
        self.fitness_function = fitness_function
        self.minimization = minimization
        self.dimensionality = len(search_space)


    def evaluate(self, solution):
        pass


    def _validate(self, solution):
        pass


    def sample_search_space(self, random_state):
        pass
