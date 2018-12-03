import logging
import numpy as np
from functools import reduce
import utils as uls
import random
from algorithms.genetic_algorithm import GeneticAlgorithm
from random_search import RandomSearch
from solutions.solution import Solution
import pandas as pd
from random import randint

class GeneticAlgorithm2(GeneticAlgorithm):

    def __init__(self, problem_instance, random_state, population_size,
                 selection, crossover, p_c, mutation, p_m, pressure):
        RandomSearch.__init__(self, problem_instance, random_state)
        self.population_size = population_size
        self.selection1 = selection
        self.selection2 = selection
        self.crossover1 = crossover
        self.crossover2 = crossover
        self.p_c2 = p_c
        self.p_c1 = p_c
        self.mutation1 = mutation
        self.mutation2 = mutation
        self.p_m = p_m
        self.p_m=p_m
        self.repetition1 = 0
        self.repetition2 = 0
        self.population_size1 = population_size/2
        self.population_size2 = population_size/2
        self.flag1 = False
        self.flag2 = False
        self.control = 2
        self.presure1=pressure
        self.presure2=pressure
        self.count = 0
        self.variation1 = 1
        self.variation2 = 1

    def initialize(self):
        self.population1 = self.generate_random_valid_solutions2()
        self.population2 = self.generate_random_valid_solutions2()
        self.population = np.concatenate((self.population1, self.population2)).tolist()
        self.best_solution1, self.best_solution2, self.best_solution = self._get_elite2(self.population1, self.population2)


    def search(self, n_iterations, report=False, log=False):

        iteration =0
        if log:
            log_event = [self.problem_instance.__class__, id(self._random_state), __name__]
            logger = logging.getLogger(','.join(list(map(str, log_event))))

        elite1 = self.best_solution1
        elite2 = self.best_solution2
        self.repetition=0

        offsprings1 = []
        offsprings2 = []


        while len(offsprings1) < len(self.population1):
            off11, off12 = p11, p12 = [
                self.selection1(self.population1, self.problem_instance.minimization, self._random_state, self.presure1)
                for _ in
                range(2)]

            if self._random_state.uniform() < self.p_c1:
                off11 = self._crossover1(p11, p12)

            if self._random_state.uniform() < self.p_m:
                # print(off11)
                off11 = self._mutation1(off11)
                off12 = self._mutation1(off12)
            if not (hasattr(off11, 'fitness') and hasattr(off12, 'fitness')):
                self.problem_instance.evaluate(off11)
                self.problem_instance.evaluate(off12)
            offsprings1.extend([off11, off12])

        while len(offsprings2) < len(self.population2):
            off21, off22 = p21, p22 = [
                self.selection2(self.population2, self.problem_instance.minimization, self._random_state,
                                self.presure2) for _ in range(2)]

            if self._random_state.uniform() < self.p_c2:
                off21 = self._crossover2(p21, p22)

            if self._random_state.uniform() < self.p_m:
                off21 = self._mutation2(off21)
                #off22 = self._mutation2(off22)

            if not (hasattr(off21, 'fitness') and hasattr(off22, 'fitness')):
                self.problem_instance.evaluate(off21)
               # self.problem_instance.evaluate(off22)
            offsprings2.extend([off21])

        while len(offsprings1) > len(self.population1):
            offsprings1.pop()

        while len(offsprings2) > len(self.population2):
            offsprings2.pop()

        elite_offspring1 = self._get_elite(offsprings1)

        elite1 = self._get_best(elite1, elite_offspring1)
        if report:
            self._verbose_reporter_inner(elite1, iteration)

        elite_offspring2 = self._get_elite(offsprings2)

        elite2 = self._get_best(elite2, elite_offspring2)
        if report:
            self._verbose_reporter_inner(elite2, iteration)

        self.variation1 = self._phenotypic_diversity_shift1(offsprings1, self.population1)
        self.variation2 = self._phenotypic_diversity_shift1(offsprings2, self.population2)

        selectTheBestOffsprings1 = pd.DataFrame(np.asanyarray([[offspring, offspring.fitness] for offspring
                                                               in offsprings1]))
        selectTheBestPopulation1 = pd.DataFrame(np.asanyarray([[individual, individual.fitness] for individual
                                                               in self.population1]))

        selectTheBestOffsprings1.rename(index=str, columns={0: "Offspring", 1: "Fitness"}, inplace=True)
        selectTheBestPopulation1.rename(index=str, columns={0: "Individual", 1: "Fitness"}, inplace=True)

        selectTheBestOffsprings1.sort_values(ascending=False, inplace=True, by="Fitness")
        selectTheBestPopulation1.sort_values(ascending=False, inplace=True, by="Fitness")

        selectTheBestOffspringsList1 = selectTheBestOffsprings1['Offspring'].tolist()
        selectTheBestPopulationList1 = selectTheBestPopulation1['Individual'].tolist()
        newPopulation1 = selectTheBestOffspringsList1[:12] + selectTheBestPopulationList1[:1]


        selectTheBestOffsprings2 = pd.DataFrame(np.asanyarray([[offspring, offspring.fitness] for offspring
                                                               in offsprings2]))
        selectTheBestPopulation2 = pd.DataFrame(np.asanyarray([[individual, individual.fitness] for individual
                                                               in self.population2]))

        selectTheBestOffsprings2.rename(index=str, columns={0: "Offspring", 1: "Fitness"}, inplace=True)
        selectTheBestPopulation2.rename(index=str, columns={0: "Individual", 1: "Fitness"}, inplace=True)

        selectTheBestOffsprings2.sort_values(ascending=False, inplace=True, by="Fitness")
        selectTheBestPopulation2.sort_values(ascending=False, inplace=True, by="Fitness")

        selectTheBestOffspringsList2 = selectTheBestOffsprings2['Offspring'].tolist()
        selectTheBestPopulationList2 = selectTheBestPopulation2['Individual'].tolist()

        newPopulation2 = selectTheBestOffspringsList2[:11] + selectTheBestPopulationList2[:1]

        self.population1 = newPopulation1
        self.population2 = newPopulation2
        self.population = []
        for solution in self.population1:
            self.population.append(solution)
        for solution in self.population2:
            self.population.append(solution)

        if elite1.fitness > self.best_solution1.fitness:
            self.best_solution1 = elite1

        if elite2.fitness > self.best_solution2.fitness:
            self.best_solution2 = elite2

        if self.best_solution2.fitness < self.best_solution1.fitness:
            self.best_solution = self.best_solution1
        else:
            self.best_solution = self.best_solution2


    def generate_random_valid_solutions2(self):
        solutions1 = np.array([self._generate_random_valid_solution()
                              for i in range(int(self.population_size1))])
        return solutions1

    def _generate_random_valid_solutions(self):
        solutions = np.array([self._generate_random_valid_solution()
                              for i in range(self.population_size)])
        return solutions


    def _get_elite2(self, population1, population2):
        elite1 = reduce(self._get_best, population1)
        elite2 = reduce(self._get_best, population2)
        eliteFinal = elite1
        if elite1.fitness < elite2.fitness:
            eliteFinal = elite2
        return elite1, elite2, eliteFinal


    def _crossover1(self, p1, p2):
        off11 = self.crossover2(p1.representation, p2.representation, self._random_state)
        off11 = Solution(off11)
        return off11

    def _crossover2(self, p1, p2):
        off1 = self.crossover2(p1.representation, p2.representation, self._random_state)
        off1 = Solution(off1)
        return off1

    def _mutation1(self, individual):
        mutant1 = self.mutation1(individual.representation, self._random_state)
        mutant1 = Solution(mutant1)
        return mutant1

    def _mutation2(self, individual):
        mutant2 = self.mutation2(individual.representation, self._random_state)
        mutant2 = Solution(mutant2)
        return mutant2

    def migration(self, solutions):
        replace = []
        for i in range(6):
            replace.append(self.population[randint(0,len(self.population)-1)])
            self.population1[randint(0,len(self.population)-1)] = solutions[i]
            self.population2[randint(0,len(self.population)-1)] = solutions[i]
        return replace

    def _phenotypic_diversity_shift1(self, offsprings, population):
        fitness_parents = np.array([parent.fitness for parent in population])
        fitness_offsprings = np.array([offspring.fitness for offspring in offsprings])
        return np.std(fitness_offsprings) - np.std(fitness_parents)

    def set_populacao1(self):
        self.population1 = self.generate_random_valid_solutions2()

    def set_populacao2(self):
        self.population2 = self.generate_random_valid_solutions2()

    def get_best_solution1(self):
        return self.best_solution1

    def get_best_solution2(self):
        return self.best_solution2
