from functools import reduce
import numpy as np
import random
from random import randint
import pandas as pd
import math

def get_random_state(seed):
    return np.random.RandomState(seed)


def random_boolean_1D_array(length, random_state):
    return random_state.choice([True, False], length)


def bit_flip(bit_string, random_state):
    neighbour = bit_string.copy()
    index = random_state.randint(0, len(neighbour))
    neighbour[index] = not neighbour[index]

    return neighbour


def parametrized_iterative_bit_flip(prob):
    def iterative_bit_flip(bit_string, random_state):
        neighbor = bit_string.copy()
        for index in range(len(neighbor)):
            if random_state.uniform() < prob:
                neighbor[index] = not neighbor[index]
        return neighbor

    return iterative_bit_flip


def random_float_1D_array(hypercube, random_state):
    return np.array([random_state.uniform(tuple_[0], tuple_[1])
                     for tuple_ in hypercube])


def random_float_cbound_1D_array(dimensions, l_cbound, u_cbound, random_state):
    return random_state.uniform(lower=l_cbound, upper=u_cbound, size=dimensions)


def parametrized_ball_mutation(radius):
    def ball_mutation(point, random_state):
        return np.array([random_state.uniform(low=coordinate - radius, high=coordinate + radius) for coordinate in point])
    return ball_mutation

#Bad
def parametrized_scramble_mutation(radius):
    def scramble_mutation(point, random_state):
       # print("Comprimento: ",len(point))

        random_number1 = random_state.randint(0, len(point)-1)
        random_number2 = random_state.randint(0, len(point))
        #print("RAndom number1: ", random_number1)
        #print("Random number2: ", random_number2)
        np.random.shuffle(point[random_number1:random_number2])
        #point = np.concatenate((point[:random_number1], point2[:], point[random_number2:]))
        return point
    return scramble_mutation

#Bad
def parametrized_swap(radius):
    def swap(point, random_state):
        point[random_state.randint(0, len(point) - 1)], \
        point[random_state.randint(0, len(point) - 1)] = \
            point[random_state.randint(0, len(point) - 1)],point[random_state.randint(0, len(point) - 1)]
        return point
    return swap


def sphere_function(point):
    return np.sum(np.power(point, 2.), axis=len(point.shape) % 2 - 1)

#Dont use
def one_point_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    point = random_state.randint(len_)
    off1_r = np.concatenate((p1_r[0:point], p2_r[point:len_]))
    off2_r = np.concatenate((p2_r[0:point], p1_r[point:len_]))
    return off1_r, off2_r

# Is kind of good. Maybe we need more points
def two_point_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    point1 = random_state.randint(len_-1)
    point2 = random_state.randint(point1+1, len_)
    off1_r = np.concatenate((p1_r[0:point1], p2_r[point1:point2], p1_r[point2: len_]))
    off2_r = np.concatenate((p2_r[0:point1], p1_r[point1:point2], p2_r[point2: len_]))
    return off1_r, off2_r

#dont remmenber
def mix_one_and_two_point(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    point1 = random_state.randint(len_)
    point2 = random_state.randint(len_)
    off1_r = np.concatenate((p1_r[0:point1], p2_r[point1:point2], p1_r[point2: len_]))
    off2_r = np.concatenate((p2_r[0:point1], p1_r[point1:point2], p2_r[point2: len_]))
    return off1_r, off2_r

#Similar with two points, but a little worst
def subtoru_exchange_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    point1 = random_state.randint(len_)
    point2 = random_state.randint(point1 + 1, len_)
    len_2 = point2-point1
    point3 = random_state.randint(len_-len_2)
    off1_r = np.concatenate((p1_r[0:point1], p2_r[point3:point3+len_2], p1_r[point2: len_]))
    off2_r = np.concatenate((p2_r[0:point3], p1_r[point1:point2], p2_r[point3+len_2: len_]))
    return off1_r, off2_r

#Dont use this
def order_crossover_random_genes(p1_r, p2_r, random_state, pressure=.05):
    len_ = len(p1_r)
    off1_r, off2_r = p1_r, p2_r
    point1 = random_state.randint(len_-1)
    point2 = random_state.randint(point1+1, len_)
    if point1!=0:
        initial_position = 0
        for i in range(point1):
            for j in range(initial_position, len_):
                value = p2_r[j]
                if value-pressure < off1_r[i] < value+pressure:
                    off1_r[i]=value
                    break
            initial_position=initial_position+1
        for i in range(point2, len_):
            for j in range(initial_position, len_):
                value= p2_r[j]
                if value-pressure < off1_r[i] < value+pressure:
                    off1_r[i]=value
                    break
            initial_position=initial_position+1

        for i in range(point1):
            for j in range(initial_position, len_):
                value = p1_r[j]
                if not value-pressure < off2_r[i] < value+pressure:
                    off2_r[i]=value
                    break
            initial_position=initial_position+1
        for i in range(point2, len_):
            for j in range(initial_position, len_):
                value= p1_r[j]
                if not value-pressure < off2_r[i] < value+pressure:
                    off2_r[i]=value
                    break
            initial_position=initial_position+1
    return off1_r, off2_r

#Bad one.
def aritmetic_recombination(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    off1_r, off2_r = p1_r, p2_r
    for iteration in range(len_):
        total_cromossome =  p1_r[iteration]+p2_r[iteration]
        if total_cromossome > 2:
            off1_r[iteration] = 2
            off2_r[iteration] = 2
        elif total_cromossome <-2:
            off1_r[iteration]= -2
            off2_r[iteration] = -2
        else:
            off1_r[iteration] = p1_r[iteration]+p2_r[iteration]
            off2_r[iteration] = p1_r[iteration]+p2_r[iteration]

    return off1_r, off2_r

#Better crossover so far
def geometric_semantic_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    off1_r = p1_r
    for iteration in range(len_):
        random = random_state.uniform(0,1)
        off1_r[iteration] = random*p1_r[iteration] + (1-random)*p2_r[iteration]
    return off1_r

#Not use. Is from the Leonardo code.
def geometric_semantic_crossover_sigmoid(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    off1_r = p1_r
    for iteration in range(len_):
        random = random_state.uniform(0,1)
        sigmoid = 1/(1+math.exp(-(random)))
        off1_r[iteration] = sigmoid*p1_r[iteration] + (1-sigmoid)*p2_r[iteration]
    return off1_r

#Very disruptive
def uniform_points_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    off1_r, off2_r = p1_r, p2_r
    for iteration in range(len_):
        random = random_state.randint(0,2)
        if random == 1:
            off1_r[iteration] = p1_r[iteration]
            off2_r[iteration] = p2_r[iteration]
        else:
            off1_r[iteration] =  p2_r[iteration]
            off2_r[iteration] =  p1_r[iteration]
    return off1_r, off2_r


def generate_cbound_hypervolume(dimensions, l_cbound, u_cbound):
  return [(l_cbound, u_cbound) for _ in range(dimensions)]


def parametrized_ann(ann_i):
  def ann_ff(weights):
    return ann_i.stimulate(weights)
  return ann_ff

#Need to confirm this. Somethimes the sames individuals are selected and is annoying.
def parametrized_tournament_selection(pressure2):
    def tournament_selection(population, minimization, random_state, pressure):
        tournament_pool_size = int(len(population)*pressure)
        tournament_pool = random_state.choice(population, size=tournament_pool_size, replace=False)
        if minimization:
            return reduce(lambda x, y: x if x.fitness <= y.fitness else y, tournament_pool)
        else:
            return reduce(lambda x, y: x if x.fitness >= y.fitness else y, tournament_pool)
    return tournament_selection


#Same.
def parametrize_roulette_wheel(pressure):
    def roulette_wheel(population, minimization, random_state, pressure):
        solution1 = None
        sum = 0

        selectTheBestPopulation1 = pd.DataFrame(
            np.asanyarray([[individual, individual.fitness] for individual
                           in population]))
        selectTheBestPopulation1.rename(index=str, columns={0: "Individual", 1: "Fitness"}, inplace=True)
        selectTheBestPopulation1.sort_values(ascending=False, inplace=True, by="Fitness")
        selectTheBestPopulationList1 = selectTheBestPopulation1['Individual'].tolist()

        i = 0

        while i < len(selectTheBestPopulationList1):
            if i < len(selectTheBestPopulationList1)/2:
                sum = sum + selectTheBestPopulationList1[i].fitness
            else:
                sum = sum + (selectTheBestPopulationList1[i].fitness*pressure)
            i = i + 1
        individual_fitness = 0
        i = 0
        while i < len(selectTheBestPopulationList1):
            if i < len(selectTheBestPopulationList1):
                individual_fitness = individual_fitness+selectTheBestPopulationList1[i].fitness
            else:
                individual_fitness = individual_fitness + (selectTheBestPopulationList1[i].fitness*pressure)
            if individual_fitness/sum >= random_state.uniform(0, 1):
                solution1 = selectTheBestPopulationList1[i]
                break
        return solution1
    return roulette_wheel

#Same 2
def parametrize_roulette_wheel2(pressure):
    def roulette_wheel2(population, minimization, random_state, pressure):
        solution1 = None
        sum = 0
        selectTheBestPopulation1 = pd.DataFrame(
            np.asanyarray([[individual, individual.fitness] for individual
                           in population]))
        selectTheBestPopulation1.rename(index=str, columns={0: "Individual", 1: "Fitness"}, inplace=True)
        selectTheBestPopulation1.sort_values(ascending=False, inplace=True, by="Fitness")
        selectTheBestPopulationList1 = selectTheBestPopulation1['Individual'].tolist()
        i = 0
        while i < len(selectTheBestPopulationList1):
            sum = sum + selectTheBestPopulationList1[i].fitness
            i = i + 1
        individual_fitness = 0
        i = 0
        while i < len(selectTheBestPopulationList1):
            individual_fitness = individual_fitness+selectTheBestPopulationList1[i].fitness
            if individual_fitness/sum >= random_state.uniform(0, 1):
                solution1 = selectTheBestPopulationList1[i]
                break
        return solution1
    return roulette_wheel2


#Need to Implement
def parametrize_rank_selection (pressure):
    def rank_selection(population, minimization, random_state, pressure):

        len_ = len(population)




    return rank_selection



def calculate_sum_solution(population):
   sum = 0
   for individual in population:
       sum = sum + individual.fitness
   return sum

def calculate_max_solution(population):
    max = population[0]
    for individual in population:
        if individual.fitness > max.fitness:
            max = individual
    return max

def calculate_min_solution(population):
    min = population[0]
    for individual in population:
        if individual.fitness < min.fitness:
            min = individual
    return min

def calculate_media_solution(population):
    return calculate_sum_solution(population)/len(population)

#Bad
def parametrize_uniform_mutation(radius):
    def uniform_points_mutation(solution, random_state):
       len_ = len(solution)
       mutant=solution
       for iteration in range(len_):
           random = random_state.randint(0, 2)
           if random==0:
               mutant[iteration] = mutant[iteration] + radius
           else:
               mutant[iteration] = mutant[iteration]-radius
       return mutant
    return uniform_points_mutation

#Bad. But we can use to generate the mutant ugly guy
def parametrize_inverse_mutation(radius):
    def inverse_mutation(solution, random_state):
       len_ = len(solution)
       point1 = random_state.randint(len_-1)
       point2 = random_state.randint(point1+1, len_)
       mutant = solution
       for iteration in range(point1):
           if mutant[iteration]<0:
               mutant[iteration]= mutant[iteration]+radius
           else:
               mutant[iteration]=mutant[iteration]-radius
       for iteration in range(point2, len_):
           if mutant[iteration] < 0:
               mutant[iteration] = mutant[iteration] + radius
           else:
               mutant[iteration] = mutant[iteration] - radius
       return mutant
    return inverse_mutation


def order_numpy_solutions_array(population):

    selectTheBestPopulation = pd.DataFrame(np.asanyarray([[individual, individual.fitness] for individual
                                                           in population]))
    selectTheBestPopulation.rename(index=str, columns={0: "Individual", 1: "Fitness"}, inplace=True)
    selectTheBestPopulation.sort_values(ascending=False, inplace=True, by="Fitness")
    selectTheBestOffspringsList2 = selectTheBestPopulation['Individual'].tolist()

    return selectTheBestOffspringsList2

def inversed_order_numpy_solutions_array(population):

    selectTheBestPopulation = pd.DataFrame(np.asanyarray([[individual, individual.fitness] for individual
                                                           in population]))
    selectTheBestPopulation.rename(index=str, columns={0: "Individual", 1: "Fitness"}, inplace=True)
    selectTheBestPopulation.sort_values(ascending=True, inplace=True, by="Fitness")
    selectTheBestOffspringsList2 = selectTheBestPopulation['Individual'].tolist()

    return selectTheBestOffspringsList2



#Not quite good, but I think the idea need to be develop
def parametrize_botzmann_selection(iteration):
    def botzmann_selection(population, minimization, random_state, pressure):
        solution1 = None
        sum = 0
        selectTheBestPopulation1 = pd.DataFrame(
            np.asanyarray([[individual, individual.fitness] for individual
                           in population]))
        selectTheBestPopulation1.rename(index=str, columns={0: "Individual", 1: "Fitness"}, inplace=True)
        selectTheBestPopulation1.sort_values(ascending=False, inplace=True, by="Fitness")
        selectTheBestPopulationList1 = selectTheBestPopulation1['Individual'].tolist()

        i = 0

        while i < len(selectTheBestPopulationList1):
            sum = sum + selectTheBestPopulationList1[i].fitness
            i = i + 1

        i = 0
        while i < len(selectTheBestPopulationList1):
            expected_value = math.exp((population[i].fitness))/(calculate_media_solution(population))
            print(expected_value)
            if random_state.uniform(0,1) < expected_value:
                solution1 = selectTheBestPopulationList1[i]
                break
            i=i+1
        return solution1
    return botzmann_selection






