class Solution:
    _id = 0

    def __init__(self, representation):
        self._solution_id = Solution._id
        Solution._id += 1
        self.representation = representation

    def print_(self, inline=True, show_list=False):
        if show_list:
            if inline:
                print(*self.representation, sep=" | ")
            else:
                print(*self.representation, sep="\n")
        print("Solution ID: %d\nDimensionality: %d\nFitness: %.2f\nIs admissible?\tR: %s" %
              (self._solution_id, self.dimensionality, self.fitness, self.valid))
