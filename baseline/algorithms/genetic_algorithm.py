import logging
import numpy as np
from functools import reduce
import utils as uls
import random

from random_search import RandomSearch
from solutions.solution import Solution
import pandas as pd

class GeneticAlgorithm(RandomSearch):


    def __init__(self, problem_instance, random_state, population_size,
                 selection, crossover, p_c, mutation, p_m):
        RandomSearch.__init__(self, problem_instance, random_state)
        self.population_size = population_size
        self.selection = selection
        self.selection2 = selection
        self.crossover = crossover
        self.p_c = p_c
        self.mutation = mutation
        self.p_m = p_m
        self.static = 0
        self.repetition = 0



    def initialize(self):
        self.population = self._generate_random_valid_solutions()
        self.best_solution = self._get_elite(self.population)


    def search(self, n_iterations, report=False, log=False):

        if log:
            log_event = [self.problem_instance.__class__, id(self._random_state), __name__]
            logger = logging.getLogger(','.join(list(map(str, log_event))))

        elite = self.best_solution
        count = 0
        for iteration in range(n_iterations):
            count = count+1
            offsprings = []
            ''' 
          
            '''
            if self.static < 0.3:
                fitness_max = uls.calculate_max_solution(self.population).fitness
                fitness_min = uls.calculate_min_solution(self.population).fitness
                fitness_avg = uls.calculate_sum_solution(self.population) / len(self.population)

                w = (((fitness_max - fitness_avg) / (fitness_max - fitness_min))
                     / self.static) ** 2
                replace = int(len(self.population)*(0.35+(random.uniform(0,1) ** w)*0.6))
                selectTheBestPopulation = pd.DataFrame(np.asanyarray([[individual, individual.fitness] for individual
                                                                      in self.population]))
                selectTheBestPopulation.rename(index=str, columns={0: "Individual", 1: "Fitness"}, inplace=True)
                selectTheBestPopulation.sort_values(ascending=False, inplace=True, by="Fitness")
                selectTheBestPopulationList = selectTheBestPopulation['Individual'].tolist()
                self.population = self._generate_random_valid_solutions().tolist()[:25] + \
                                  selectTheBestPopulationList[:len(self.population)-25]


            while len(offsprings) < len(self.population):
                off1, off2 = p1, p2 = [
                    self.selection(self.population, self.problem_instance.minimization, self._random_state) for _ in range(2)]
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
            if report:
                self._verbose_reporter_inner(elite, iteration)



            if log:
                log_event = [iteration, elite.fitness, elite.validation_fitness if hasattr(off2, 'validation_fitness') else None,
                             self.population_size, self.selection.__name__, self.crossover.__name__, self.p_c,
                             self.mutation.__name__, None, None, self.p_m, self._phenotypic_diversity_shift(offsprings)]
                logger.info(','.join(list(map(str, log_event))))

            print(self._phenotypic_diversity_shift(offsprings))

            if self._phenotypic_diversity_shift(offsprings) < 0.07:
                self.repetition = self.repetition+1

            #Elitism
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

            fitness_max = uls.calculate_max_solution(newPopulation).fitness
            fitness_min = uls.calculate_min_solution(newPopulation).fitness
            fitness_avg = uls.calculate_sum_solution(newPopulation)/len(self.population)
          #  print(self._phenotypic_diversity_shift(offsprings))
          # print("soma: ", uls.calculate_sum_solution(self.population))
            print("avg: ", uls.calculate_sum_solution(newPopulation)/len(self.population))
          #  print("min: ", uls.calculate_min_solution(self.population).fitness)
            print("max: ", uls.calculate_max_solution(newPopulation).fitness)
            print('Comprimento da população: ',len(self.population))
          #  print("fitness da população: ",  uls.calculate_sum_solution(self.population)/len(self.population))
            self.static = (fitness_max - fitness_avg) / (fitness_max-fitness_min)
            print("Diversidade-População: ", (fitness_max - fitness_avg) / (fitness_max-fitness_min))
            print('Comprimento dos offsprings: ', len(self.population))
          #  print("fitness dos offsprincs: ", uls.calculate_sum_solution(newPopulation)/len(newPopulation))
            print("MAX-Population: ", uls.calculate_max_solution(newPopulation).fitness)
            print("MAX-Offsprings: ", uls.calculate_max_solution(offsprings).fitness)

            self.best_solution = elite

            self.population = newPopulation

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
        return np.std(fitness_offsprings)-np.std(fitness_parents)

    def _generate_random_valid_solutions(self):
        solutions = np.array([self._generate_random_valid_solution()
                              for i in range(self.population_size)])
        return solutions

