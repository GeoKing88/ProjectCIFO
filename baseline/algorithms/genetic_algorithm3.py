import logging
import numpy as np
from functools import reduce
import utils as uls
import random
from algorithms.genetic_algorithm import GeneticAlgorithm
from random_search import RandomSearch
from solutions.solution import Solution
import pandas as pd


class GeneticAlgorithm3(GeneticAlgorithm):

    def __init__(self, problem_instance, random_state, population_size,
                 selection, crossover, p_c, mutation, p_m):
        RandomSearch.__init__(self, problem_instance, random_state)
        self.population_size = population_size
        self.selection1 = selection
        self.selection2 = selection
        self.crossover1 = crossover
        self.crossover2 = crossover
        self.crossover3 = crossover
        self.p_c2 = p_c
        self.p_c1 = p_c
        self.pc_c3 = p_c
        self.mutation = mutation
        self.p_m1 = p_m
        self.p_m2=p_m
        self.p_m3=p_m
        self.static1 = 0
        self.static2 = 0
        self.static3 = 0
        self.repetition1 = 0
        self.repetition2 = 0
        self.repetition3 = 0
        self.population_size1 = population_size/2
        self.population_size2 = population_size/2
        self.population_size3 = population_size/2
        self.flag1 = False
        self.flag2 = False
        self.flag3 = False

    def initialize(self):
        self.population1, self.population2, self.population3 = self._generate_random_valid_solutions3()
        self.population = np.concatenate((self.population1, self.population2, self.population3))
        self.best_solution1, self.best_solution2, self.best_solution3, \
            self.best_solution = self._get_elite3(self.population1, self.population2, self.population3)


    def search(self, n_iterations, report=False, log=False):

        if log:
            log_event = [self.problem_instance.__class__, id(self._random_state), __name__]
            logger = logging.getLogger(','.join(list(map(str, log_event))))

        elite1 = self.best_solution1
        elite2 = self.best_solution2
        elite3 = self.best_solution3

        for iteration in range(n_iterations):

            offsprings1 = []
            offsprings2 = []
            offspring3 = []

            if self.repetition1>2 and self.flag1:
                print(">>>>>>>>>>>>>>>>>>>>>>>>> TRUE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                self.p_c1=0.9
                self.selection1 = uls.parametrized_tournament_selection(0.1)
                #self.selection2 = uls.parametrize_roulette_wheel()
                self.p_m1 = 0.6
                self.repetition1 = 0
                self.flag1 = False

            if self.repetition1>2 and self.flag1==False:
                print(">>>>>>>>>>>>>>>>>>>>>>>>> TRUE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                self.p_c1 = 0.9
                self.p_m1 = 0.6
                self.selection1 = uls.parametrize_roulette_wheel()
                #self.selection2 = uls.parametrized_tournament_selection(0.1)
                self.repetition1=0
                self.flag1=True

            if self.repetition2>2 and self.flag2:
                print(">>>>>>>>>>>>>>>>>>>>>>>>> TRUE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                self.p_c2=0.8
                self.selection = uls.parametrized_tournament_selection(0.1)
                #self.selection2 = uls.parametrize_roulette_wheel()
                self.p_m2 = 0.3
                self.repetition2 = 0
                self.flag2 = False

            if self.repetition2>2 and self.flag2==False:
                print(">>>>>>>>>>>>>>>>>>>>>>>>> TRUE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                self.p_c2 = 0.8
                self.p_m2 = 0.3
                self.selection = uls.parametrize_roulette_wheel()
                #self.selection2 = uls.parametrized_tournament_selection(0.1)
                self.repetition2=0
                self.flag2=True

            if self.repetition3 > 2 and self.flag3:
                print(">>>>>>>>>>>>>>>>>>>>>>>>> TRUE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                self.p_c3 = 0.8
                self.selection = uls.parametrized_tournament_selection(0.1)
                # self.selection2 = uls.parametrize_roulette_wheel()
                self.p_m3 = 0.3
                self.repetition3 = 0
                self.flag3 = False

            if self.repetition3 > 2 and self.flag3 == False:
                print(">>>>>>>>>>>>>>>>>>>>>>>>> TRUE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                self.p_c3 = 0.8
                self.p_m3 = 0.3
                self.selection = uls.parametrize_roulette_wheel()
                # self.selection2 = uls.parametrized_tournament_selection(0.1)
                self.repetition3 = 0
                self.flag3 = True

            if self.static1 < 0.35:

                self.selection1 = uls.parametrize_ranking()

            else:
                self.selection1=uls.parametrized_tournament_selection(0.1)

            if iteration ==30 or iteration==60 or iteration==90:
                    fitness_max = uls.calculate_max_solution(self.population).fitness
                    fitness_min = uls.calculate_min_solution(self.population).fitness
                    fitness_avg = uls.calculate_sum_solution(self.population) / len(self.population)

                    w = (((fitness_max - fitness_avg) / (fitness_max - fitness_min))
                         / self.static1) ** 2
                    replace = int(len(self.population) * (0.35 + (random.uniform(0, 1) ** w) * 0.6))
                    selectTheBestPopulation1 = pd.DataFrame(
                        np.asanyarray([[individual, individual.fitness] for individual
                                       in self.population1]))
                    selectTheBestPopulation1.rename(index=str, columns={0: "Individual", 1: "Fitness"}, inplace=True)
                    selectTheBestPopulation1.sort_values(ascending=False, inplace=True, by="Fitness")
                    selectTheBestPopulationList1 = selectTheBestPopulation1['Individual'].tolist()

                    selectTheBestPopulation2 = pd.DataFrame(
                        np.asanyarray([[individual, individual.fitness] for individual
                                       in self.population2]))
                    selectTheBestPopulation2.rename(index=str, columns={0: "Individual", 1: "Fitness"}, inplace=True)
                    selectTheBestPopulation2.sort_values(ascending=False, inplace=True, by="Fitness")
                    selectTheBestPopulationList2 = selectTheBestPopulation2['Individual'].tolist()

                    self.population1 = selectTheBestPopulationList1[:len(self.population1) - replace] + \
                                       selectTheBestPopulationList2[:replace]

                    self.population2 = selectTheBestPopulationList2[:len(self.population1) - replace] + \
                                       selectTheBestPopulationList1[:replace]

            if self.static2 < 0.35:
                self.selection2 = uls.parametrize_ranking()
            else:
                self.selection2 = uls.parametrized_tournament_selection(0.1)




            while len(offsprings1) < len(self.population1):
                off11, off12 = p11, p12 = [
                    self.selection1(self.population1, self.problem_instance.minimization, self._random_state) for _ in range(2)]

                if self._random_state.uniform() < self.p_c1:
                    off11, off12 = self._crossover1(p11, p12)

                if self._random_state.uniform() < self.p_m1:
                    off11 = self._mutation(off11)
                    off12= self._mutation(off12)

                if not (hasattr(off11, 'fitness') and hasattr(off12, 'fitness')):
                    self.problem_instance.evaluate(off11)
                    self.problem_instance.evaluate(off12)
                offsprings1.extend([off11, off12])



            while len(offsprings2) < len(self.population2):
                off21, off22 = p21, p22 = [
                    self.selection2(self.population2, self.problem_instance.minimization, self._random_state) for _ in range(2)]

                if self._random_state.uniform() < self.p_c2:
                    off21, off22 = self._crossover2(p21, p22)

                if self._random_state.uniform() < self.p_m2:
                    off21 = self._mutation(off21)
                    off22= self._mutation(off22)

                if not (hasattr(off21, 'fitness') and hasattr(off22, 'fitness')):
                    self.problem_instance.evaluate(off21)
                    self.problem_instance.evaluate(off22)
                offsprings2.extend([off21, off22])

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

            print('_phenotypic_diversity_shift - offspring1 ', self._phenotypic_diversity_shift(offsprings1))
            print('_phenotypic_diversity_shift - offspring2 ', self._phenotypic_diversity_shift(offsprings2))

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

            newPopulation1 = selectTheBestOffspringsList1[:15] + selectTheBestPopulationList1[:5] +\
                             selectTheBestPopulationList1[-5:]

            fitness_max1 = uls.calculate_max_solution(newPopulation1).fitness
            fitness_min1 = uls.calculate_min_solution(newPopulation1).fitness
            fitness_avg1 = uls.calculate_sum_solution(newPopulation1) / len(self.population1)
            #  print(self._phenotypic_diversity_shift(offsprings))
            # print("soma: ", uls.calculate_sum_solution(self.population))
            print("Comprimento 1: ", len(self.population1))
            print("avg POP1: ", uls.calculate_sum_solution(newPopulation1) / len(self.population1))
            print("max: ", uls.calculate_max_solution(newPopulation1).fitness)
            print("\n")
            self.static1 = (fitness_max1 - fitness_avg1) / (fitness_max1 - fitness_min1)
            #  print("fitness dos offsprincs: ", uls.calculate_sum_solution(newPopulation)/len(newPopulation))
           # print("MAX-Population1: ", uls.calculate_max_solution(newPopulation1).fitness)
           # print("MAX-Offsprings1: ", uls.calculate_max_solution(offsprings1).fitness)

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


            newPopulation2 = selectTheBestOffspringsList2[:15] + selectTheBestPopulationList2[:5]+\
                             selectTheBestOffspringsList2[-5:]
            fitness_max2 = uls.calculate_max_solution(newPopulation2).fitness
            fitness_min2 = uls.calculate_min_solution(newPopulation2).fitness
            fitness_avg2 = uls.calculate_sum_solution(newPopulation2) / len(self.population2)
            #  print(self._phenotypic_diversity_shift(offsprings))
            # print("soma: ", uls.calculate_sum_solution(self.population))
            print("avg POP2: ", uls.calculate_sum_solution(newPopulation2) / len(self.population2))
            self.static2 = (fitness_max2 - fitness_avg2) / (fitness_max2 - fitness_min2)
            print("Comprimento 2: ", len(self.population2))
            #  print("fitness dos offsprincs: ", uls.calculate_sum_solution(newPopulation)/len(newPopulation))
            print("MAX-Population2: ", uls.calculate_max_solution(newPopulation2).fitness)
          #  print("MAX-Offsprings2: ", uls.calculate_max_solution(offsprings2).fitness)
           # print("max: ", uls.calculate_max_solution(newPopulation2).fitness)


            if self.best_solution1.fitness == elite1.fitness:
                self.repetition1 = self.repetition1+1
            else:
                self.repetition1=0



            if self.best_solution2.fitness == elite2.fitness:
                self.repetition2 = self.repetition2+1
            else:
                self.repetition2=0

            if elite1.fitness > elite2.fitness:
                eliteTotal = elite1
            else:
                eliteTotal = elite2

            if eliteTotal == self.best_solution:
                self.repetition = self.repetition + 1
            else:
                self.repetition=0
            print(">>>>>>>>>>>>>>>>>>>>> REPETITION >>>>>>>>>>>>>>>>>>: ", self.repetition)
            print(">>>>>>>>>>>>>>>>>>>>> DIVERSIDADE 1 <<<<<<<<<<<<<<<: ", self.static1)
            print(">>>>>>>>>>>>>>>>>>>>> DIVERSIDADE 1 <<<<<<<<<<<<<<<: ", self.static2)
            print(">>>>>>>>>>>>>>>>>>>>> ID >>>>>>>>>>>>>>>>>>>>>>>>>>: ", self.best_solution._solution_id)
            print(">>>>>>>>>>>>>>>>>>>>< INTERACTION <<<<<<<<<<<<< :", iteration)
            print(">>>>>>>>>>>>>>>>>>>>< FITNESS TOTAL <<<<<<<<<<<<< :" , eliteTotal.fitness)
            if elite1.fitness> elite2.fitness:
                self.best_solution = elite1
            else:
                self.best_solution = elite2

            self.population1 = newPopulation1
            self.population2 = newPopulation2
            self.population = newPopulation1 + newPopulation2
            self.best_solution1 = elite1
            self.best_solution2=elite2


    def _generate_random_valid_solutions3(self):
        solutions1 = np.array([self._generate_random_valid_solution()
                              for i in range(int(self.population_size1))])
        solutions2 = np.array([self._generate_random_valid_solution()
                               for i in range(int(self.population_size1))])
        solutions3 = np.array([self._generate_random_valid_solution()
                               for i in range(int(self.population_size1))])
        return solutions1, solutions2, solutions3




    def _get_elite3(self, population1, population2, population3):
        elite1 = reduce(self._get_best, population1)
        elite2 = reduce(self._get_best, population2)
        elite3 = reduce(self._get_best, population3)
        eliteFinal = elite1
        if elite1.fitness < elite2.fitness:
            eliteFinal = elite2
            if eliteFinal.fitness < elite3.fitness:
                eliteFinal = elite3
        else:
            if elite1.fitness < elite3.fitness:
                eliteFinal = elite3

        return elite1, elite2, elite3, eliteFinal


    def _crossover1(self, p1, p2):
        off1, off2 = self.crossover1(p1.representation, p2.representation, self._random_state)
        off1, off2 = Solution(off1), Solution(off2)
        return off1, off2

    def _crossover2(self, p1, p2):
        off1, off2 = self.crossover2(p1.representation, p2.representation, self._random_state)
        off1, off2 = Solution(off1), Solution(off2)
        return off1, off2