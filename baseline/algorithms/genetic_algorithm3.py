import logging
import numpy as np
from functools import reduce
import utils as uls
from random import randint

from random_search import RandomSearch
from solutions.solution import Solution
import pandas as pd


class GeneticAlgorithm3(RandomSearch):

    def __init__(self, problem_instance, random_state, population_size,
                 selection, crossover, p_c, mutation, p_m):
        RandomSearch.__init__(self, problem_instance, random_state)
        self.population_size = population_size
        self.selection = selection
        self.crossover = crossover
        self.p_c = p_c
        self.mutation = mutation
        self.p_m = p_m
        self.repetition = 0
        self.presure = 0.2
        self.list = []
        self.list_iteration = []

    def initialize(self):
        self.population = self._generate_random_valid_solutions()
        self.best_solution = self._get_elite(self.population)

    def search(self, n_iterations, report=False, log=False):
        elite = self.best_solution
        offsprings = []
        while len(offsprings) < len(self.population):
            off1, off2 = p1, p2 = [
                self.selection(self.population, self.problem_instance.minimization, self._random_state,
                               self.presure)
                for _ in
                range(2)]

            if self._random_state.uniform() < self.p_c:
                off1, off2 = self._crossover(p1, p2)

            if self._random_state.uniform() < self.p_m:
                off1 = self._mutation(off1)
                off2 = self._mutation(off2)

            if not (hasattr(off1, 'fitness') and hasattr(off2, 'fitness')):
                self.problem_instance.evaluate(off1)
                self.problem_instance.evaluate(off2)
            offsprings.extend([off1, off2])

        while len(offsprings) > len(self.population):
            offsprings.pop()

        elite_offspring = self._get_elite(offsprings)
        elite = self._get_best(elite, elite_offspring)

        # Elitism
        selectTheBestOffsprings = pd.DataFrame(np.asanyarray([[offspring, offspring.fitness] for offspring
                                                              in offsprings]))
        selectTheBestPopulation = pd.DataFrame(np.asanyarray([[individual, individual.fitness] for individual
                                                              in self.population]))

        selectTheBestOffsprings.rename(index=str, columns={0: "Offspring", 1: "Fitness"}, inplace=True)
        selectTheBestPopulation.rename(index=str, columns={0: "Individual", 1: "Fitness"}, inplace=True)

        selectTheBestOffsprings.sort_values(ascending=False, inplace=True, by="Fitness")
        selectTheBestPopulation.sort_values(ascending=False, inplace=True, by="Fitness")

        selectTheBestOffspringsList = selectTheBestOffsprings['Offspring'].tolist()
        selectTheBestPopulationList = selectTheBestPopulation['Individual'].tolist()

        newPopulation = selectTheBestOffspringsList[:24] + selectTheBestPopulationList[:1]

        self.population = newPopulation
        if elite.fitness == self.best_solution.fitness:
            self.repetition = self.repetition+1
        else:
            self.best_solution = elite
            self.repetition = 0

        if self.repetition > 3:
            if self.selection == uls.parametrize_roulette_wheel(self.presure):
                self.selection = uls.parametrized_tournament_selection(self.presure)
                self.repetition = 0
            else:
                self.selection = uls.parametrize_roulette_wheel(self.presure)
                self.repetition = 0

        if self.repetition >8:
            self.p_m = 0.9

        if self.p_m == 0.9 and  self.repetition<4:
            self.p_m = 0.3



    def _crossover(self, p1, p2):
        off1, off2 = self.crossover(p1.representation, p2.representation, self._random_state)
        off1, off2 = Solution(off1), Solution(off2)
        return off1, off2

    def _mutation(self, individual):
        mutant = self.mutation(individual.representation, self._random_state)
        mutant = Solution(mutant)
        return mutant

    def _get_elite(self, population):
        elite = reduce(self._get_best, population)
        return elite

    def _phenotypic_diversity_shift(self, offsprings):
        fitness_parents = np.array([parent.fitness for parent in self.population])
        fitness_offsprings = np.array([offspring.fitness for offspring in offsprings])
        return np.std(fitness_offsprings) - np.std(fitness_parents)

    def _generate_random_valid_solutions(self):
        solutions = np.array([self._generate_random_valid_solution()
                              for i in range(self.population_size)])
        return solutions

    def get_best_solution(self):
        return self.list

    def get_best_iteration(self):
        return self.list_iteration

    def migration(self, solutions):
        selectTheBestPopulation = pd.DataFrame(np.asanyarray([[individual, individual.fitness] for individual
                                                              in self.population]))
        selectTheBestPopulation.rename(index=str, columns={0: "Individual", 1: "Fitness"}, inplace=True)
        selectTheBestPopulation.sort_values(ascending=False, inplace=True, by="Fitness")
        selectTheBestPopulationList = selectTheBestPopulation['Individual'].tolist()
        for i in range(4):
            position = randint(0,len(self.population)-1)
            self.population[position] = solutions[i]
        return selectTheBestPopulationList[:4]
