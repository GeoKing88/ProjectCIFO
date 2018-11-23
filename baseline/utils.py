from functools import reduce
import numpy as np
import random
from random import randint

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
    def ball_mutation(point, random_state, radius):
        return np.array([random_state.uniform(low=coordinate - radius, high=coordinate + radius) for coordinate in point])
    return ball_mutation

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


def parametrized_swap(radius):
    def swap(point, random_state):
        point[random_state.randint(0, len(point) - 1)], \
        point[random_state.randint(0, len(point) - 1)] = \
            point[random_state.randint(0, len(point) - 1)],point[random_state.randint(0, len(point) - 1)]
        return point
    return swap


def sphere_function(point):
    return np.sum(np.power(point, 2.), axis=len(point.shape) % 2 - 1)


def one_point_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    point = random_state.randint(len_)
    off1_r = np.concatenate((p1_r[0:point], p2_r[point:len_]))
    off2_r = np.concatenate((p2_r[0:point], p1_r[point:len_]))
    return off1_r, off2_r

def two_point_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    point1 = random_state.randint(len_-1)

    point2 = random_state.randint(point1+1, len_)

    off1_r = np.concatenate((p1_r[0:point1], p2_r[point1:point2], p1_r[point2: len_]))
    off2_r = np.concatenate((p2_r[0:point1], p1_r[point1:point2], p2_r[point2: len_]))
    return off1_r, off2_r

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


def parametrized_tournament_selection(pressure):
    def tournament_selection(population, minimization, random_state):
        tournament_pool_size = int(len(population)*pressure)
        tournament_pool = random_state.choice(population, size=tournament_pool_size, replace=False)
        if minimization:
            return reduce(lambda x, y: x if x.fitness <= y.fitness else y, tournament_pool)
        else:
            return reduce(lambda x, y: x if x.fitness >= y.fitness else y, tournament_pool)
    return tournament_selection



def parametrize_roulette_wheel():
    def roulette_wheel(population, minimization, random_state):
        solution1 = None
        sum = 0
        for individual in population:
            sum = sum + individual.fitness
        individual_fitness = 0
        for individual in population:
            individual_fitness = individual_fitness+individual.fitness
            if individual_fitness/sum >= random.uniform(0, 1):
                solution1 = individual
                break
        return solution1
    return roulette_wheel

def parametrize_ranking():
    def ranking(population, minimization, random_state):
        return population[randint(0,len(population)-1)]
    return ranking


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

def parametrize_uniform_mutation(radius):
    def uniform_points_mutation(solution, random_state, radius):
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









