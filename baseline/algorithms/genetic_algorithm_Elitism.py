import logging
import numpy as np
from functools import reduce
import utils as uls
import random
import copy
from random_search import RandomSearch
from solutions.solution import Solution
import pandas as pd
import math

class GeneticAlgorithm(RandomSearch):

    def __init__(self, problem_instance, random_state, population_size,
                 selection, crossover, p_c, mutation, p_m, presure):
        RandomSearch.__init__(self, problem_instance, random_state)
        self.population_size = population_size
        self.selection = selection
        self.crossover = crossover
        self.p_c = p_c
        self.mutation = mutation
        self.p_m = p_m
        self.presure = presure
        self.reproduttive_guys = []
        self.repetition = 0

    def initialize(self):
        self.population = self._generate_random_valid_solutions()
        self.best_solution = self._get_elite(self.population)

    def search(self, n_iterations, report=False, log=False):

        if log:
            log_event = [self.problem_instance.__class__, id(self._random_state), __name__]
            logger = logging.getLogger(','.join(list(map(str, log_event))))
        elite = self.best_solution
        offsprings = []
        while len(offsprings) < len(self.population):
            off1, off2 = p1, p2 = self.selection(self.population, self._random_state, self.presure)
            if self._random_state.uniform() < self.p_c:
                off1, off2 = self._crossover(p1, p2)

            if self._random_state.uniform() < self.p_m:
                off1 = self._mutation(off1)
                off2 = self._mutation(off2)

            if not (hasattr(off1, 'fitness')):
                self.problem_instance.evaluate(off1)
                self.problem_instance.evaluate(off2)


            offsprings.append(off1)
            offsprings.append(off2)

        while len(offsprings) > len(self.population):
            offsprings.pop()

        elite_offspring = self._get_elite(offsprings)
        elite = self._get_best(elite, elite_offspring)
        self.best_solution = elite
        self.population = np.array(self.create_elites(offsprings))


    def _crossover(self, p1, p2):
        off1, off2 = self.crossover(p1.representation, p2.representation, self._random_state)
        off1 = Solution(off1)
        off2 = Solution(off2)
        return off1, off2

    def _mutation(self, individual):
        mutant = self.mutation(individual.representation, self._random_state, self.population)
        mutant = Solution(mutant)
        return mutant

    def _get_elite(self, population):
        elite = reduce(self._get_best, population)
        return elite

    def _phenotypic_diversity_shift(self, offsprings):
        fitness_parents = np.array([parent.fitness for parent in self.population])
        fitness_offsprings = np.array([offspring.fitness for offspring in offsprings])
        return np.std(fitness_offsprings)-np.std(fitness_parents)

    def _generate_random_valid_solutions(self):
        solutions = np.array([self._generate_random_valid_solution()
                              for i in range(self.population_size)])
        return solutions

    def get_all_fitness(self):
        fitness_list = []
        for i in range(len(self.population)):
            fitness_list.append(self.population[i].fitness)
        return np.asarray(fitness_list)

    def create_avg_gene_value(self):
        totais = self.best_solution.representation
        for i in range(len(self.best_solution.representation)):
            sum = 0
            for solution in self.population:
                sum = sum + solution.representation[i]
            totais[i] = sum/len(self.population)
        return totais


    def _determine_average_reproductive_guys(self):
        sum = 0
        for i in range(len(self.reproduttive_guys)):
            sum = sum + self.reproduttive_guys[i].fitness
        return sum/(len(self.reproduttive_guys))


    def determinine_pressure(self):
        print("Média: ", self._determine_average_reproductive_guys())
        print("Denominador: ", uls.calculate_media_solution(self.population))
        return (self.presure*self._determine_average_reproductive_guys())/(uls.calculate_media_solution(self.population))


    def calculate_probability_of_crossover(self, p1, p2, crossover_pressure):

        if p1.fitness < p2.fitness:
            parent_fitness = p2.fitness
        else:
            parent_fitness = p1.fitness
        avg = uls.calculate_media_solution(self.population)
        if parent_fitness >= avg:
            best_fitness = self.best_solution.fitness
            return crossover_pressure*(best_fitness - parent_fitness)/(best_fitness - avg)
        return self.p_c

    def distance_scheme(self, individual1, individual2):
        distance_map = []
        for i in range(len(individual1.representation)):
            distance_map.append(abs(individual1.representation[i] - individual2.representation[i]))
        distance = 0
        for i in range(len(distance_map)):
            distance = distance + distance_map[i]
        return distance


    def sort_populations(self, population):
        selectTheBestOffsprings = pd.DataFrame(np.asanyarray([[offspring, offspring.fitness] for offspring
                                                              in population]))
        selectTheBestOffsprings.rename(index=str, columns={0: "Offspring", 1: "Fitness"}, inplace=True)
        selectTheBestOffsprings.sort_values(ascending=False, inplace=True, by="Fitness")
        selectTheBestOffspringsList = selectTheBestOffsprings['Offspring'].tolist()
        return selectTheBestOffspringsList


    def create_elites(self, offsprings):
        population = self.sort_populations(self.population)
        offsprings = self.sort_populations(offsprings)
        return population[:1] + offsprings[:self.population_size-1]

    def commit_genocid(self, population):
        martials = self._generate_random_valid_solutions().tolist()[:12]
        population = self.sort_populations(self.population)
        return np.asanyarray(martials[:13] + population[:13])

