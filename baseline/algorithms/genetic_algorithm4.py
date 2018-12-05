import numpy as np
from functools import reduce
import utils as uls
from random_search import RandomSearch
from solutions.solution import Solution
import pandas as pd
import copy
import math

class GeneticAlgorithm(RandomSearch):


    def __init__(self, problem_instance, random_state, population_size,
                 selection, crossover, p_c, mutation, p_m, pressure):
        RandomSearch.__init__(self, problem_instance, random_state)
        self.population_size = population_size
        self.selection = selection
        self.crossover = crossover
        self.p_c = p_c
        self.mutation = mutation
        self.p_m = p_m
        self.pressure = pressure
        self.reproduttive_guys = []
        self.b = -2
        self.MAX = 2
        self.flag = True

    def initialize(self):
        self.population = self._generate_random_valid_solutions()
        self.best_solution = self._get_elite(self.population)

    def search(self, n_iterations=0, report=False, log=False):
        offsprings = []
        while len(offsprings) < len(self.population):

            off1, off2 = p1, p2 = self.selection(self.population, self._random_state
                                                 , self.pressure)


            if self._random_state.uniform() < self.p_c:
                self._crossover(p1, p2)

            if self._random_state.uniform() < self.p_m:
                off1 = self._mutation(off1)
                off2 = self._mutation(off2)

            if not (hasattr(off1, 'fitness') and hasattr(off2, 'fitness')):
                self.problem_instance.evaluate(off1)
                self.problem_instance.evaluate(off2)

            #self.select_the_offpring(p1, p2, off1, off2, offsprings)

            offsprings.extend([off1, off2])


        while len(offsprings) > len(self.population):
            offsprings.pop()


        #print(np.std([offspring.fitness for offspring in offsprings]))

        if np.std([offspring.fitness for offspring in offsprings])<0.02:
            self.selection = uls.parametrized_tournament_selection(0.2)
        else:
            self.selection = uls.parametrize_roulette_wheel_w_pressure(0.2)


        self.population = np.asarray(self.sort_populations(offsprings))
        self.best_solution = self._get_elite(self.population)

        if report:
            self._verbose_reporter_inner(self.best_solution, n_iterations)



        if self.flag:
            if self.best_solution.fitness - uls.calculate_media_solution(self.population) < 0.02:
                self.selection = uls.parametrize_rank_selection(self.pressure)
            else:
                self.selection = uls.parametrize_roulette_wheel_w_pressure(self.pressure)


        #self.distance_scheme(self.average_distance_population_normalize(), )



    def _crossover(self, p1, p2):
        off1, off2 = self.crossover(p1.representation, p2.representation, self._random_state)
        off1 = Solution(off1)
        off2 = Solution(off2)
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
        return (self.pressure * self._determine_average_reproductive_guys()) / (uls.calculate_media_solution(self.population))


    def calculate_probability_of_crossover(self, individual, crossover_pressure):

        avg = uls.calculate_media_solution(self.population)
        if individual.fitness >= avg:
            best_fitness = self.best_solution.fitness
            return crossover_pressure*(best_fitness - individual.fitness)/(best_fitness - avg)
        return self.p_c

    def distance_between_populations(self, avg_1, avg_2):
        distance_map = []
        for i in range(len(avg_1)):
            distance_map.append((avg_1[i] - avg_2[i]) ** 2)
        distance = 0
        for i in range(len(distance_map)):
            distance = distance + distance_map[i]
        return distance

    def average_distance_population(self):
        df = pd.DataFrame(columns=[i for i in range(len(self.population))])
        for i in range(len(df.columns)):
            df[i] = self.population[i].representation
        df['Mean'] = df.sum(axis=1)/len(self.population)
        return np.ravel(df['Mean'].values)


    def save_genes(self, population):
        all_genes = []
        for individual in population:
            all_genes.append(individual.representation)
        return all_genes



    def sort_populations(self, offsprings):
        selectTheBestOffsprings = pd.DataFrame(np.asanyarray([[offspring, offspring.fitness] for offspring
                                                              in offsprings]))
        selectTheBestOffsprings.rename(index=str, columns={0: "Offspring", 1: "Fitness"}, inplace=True)
        selectTheBestOffsprings.sort_values(ascending=False, inplace=True, by="Fitness")
        selectTheBestOffspringsList = selectTheBestOffsprings['Offspring'].tolist()
        selectTheBestOffspringsList.append(self.best_solution)
        return selectTheBestOffspringsList

    # A Kind of Deterministic Crowding
    def select_the_offpring(self, p1, p2, off1, off2, offsprings):

        p1_flag = False
        p2_flag = False
        if off1.fitness > p1.fitness and off1.fitness > p2.fitness:
            offsprings.append(off1)
        elif p1.fitness > p2.fitness:
            offsprings.append(p1)
            p1_flag = True
        else:
            offsprings.append(p2)
            p2_flag = True

        if off2.fitness > p1.fitness and off2.fitness > p2.fitness:
            offsprings.append(off2)
        elif p1.fitness > p2.fitness and not p1_flag:
            offsprings.append(p1)
            p1_flag = False

        if off2.fitness < p1.fitness and off2.fitness < p2.fitness and p2.fitness > p1.fitness and not p2_flag:
            offsprings.append(p2)
            p2_flag = False




