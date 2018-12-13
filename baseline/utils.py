from functools import reduce
import numpy as np
import random
from random import randint
import pandas as pd
import math
import copy
import seaborn as sbn
import matplotlib.pyplot as plt
from solution import Solution
from random import shuffle




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
    def ball_mutation(point, random_state, population):
        return np.array([random_state.uniform(low=coordinate - radius, high=coordinate + radius) for coordinate in point])
    return ball_mutation


def parametrized_gaussian_member_mutation(radius):
    def gaussian_member_mutation(point, random_state, population):
        index = random_state.randint(low=0, high=len(point), size=int(len(point)*radius))
        new_points = point.copy()
        for i in index:
            valor = random_state.normal(loc=0, scale=5)
            #print(valor)
            new_points[i] = valor
        return new_points
    return gaussian_member_mutation


def parametrized_logistic_distribution(radius):
    def logistic_distribution(point, random_state, population):
        index = random_state.randint(low=0, high=len(point), size=int(len(point) * radius))
        new_points = point.copy()
        for i in index:
            valor = random_state.logistic(loc=0, scale=10)
            #print(valor)
            new_points[i] = valor
        return new_points
    return logistic_distribution


def parametrized_laplace_mutation(radius):
    def laplace_member_mutation(point, random_state, population):
        index = random_state.randint(low=0, high=len(point), size=int(len(point)*radius))
        new_points = point.copy()
        for i in index:
            valor = random_state.laplace(loc=0, scale=2)
            new_points[i] = valor
        return new_points
    return laplace_member_mutation



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

#dont remmember
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


def partially_matched_crossover(p1_r, p2_r, random_state, pressure=1):
    point1 = random_state.randint(len(p1_r)-1)
    point2 = random_state.randint(point1+1, len(p1_r))
    index1 = []
    for i in range(len(p1_r)):
        index1.append(i)
    index2 = copy.deepcopy(index1)
    shuffle(index1)
    shuffle(index2)
    off1_r, off2_r = p1_r, p2_r

    for i in range(point1, point2+1):
        off1_r[i] = p2_r[index2[i]]
        off2_r[i] = p1_r[index1[i]]

        for j in range(len(p1_r)):
            if off1_r[j] == index2[i]:
                off1_r[j] = p2_r[i]
            if off2_r[j] == index1[i]:
                off2_r[j] = p1_r[i]

    pandas_to_sort_1 = pd.DataFrame(np.asanyarray(index1))
    pandas_to_sort_1.rename(index=str, columns={0: "Index"}, inplace=True)
    pandas_to_sort_1['Off1'] = off1_r
    pandas_to_sort_1.sort_values(ascending=False, inplace=True, by="Index")
    pandas_to_sort_1_list = pandas_to_sort_1['Off1'].tolist()

    pandas_to_sort_2 = pd.DataFrame(np.asanyarray(index2))
    pandas_to_sort_2.rename(index = str, columns={0: "Index"}, inplace=True)
    pandas_to_sort_2['Off2'] = off2_r
    pandas_to_sort_2.sort_values(ascending=False, inplace=True, by="Index")
    pandas_to_sort_2_list = pandas_to_sort_2['Off2'].tolist()
    return pandas_to_sort_1_list, pandas_to_sort_2_list



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
    off2_r = p2_r
    for iteration in range(len_):
        random = random_state.uniform(0, 1)
        off1_r[iteration] = random*p1_r[iteration] + (1-random)*p2_r[iteration]
        off2_r[iteration] = (1-random)*p1_r[iteration] + random*p2_r[iteration]
    return off1_r, off2_r

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

#Same.
def parametrize_roulette_wheel_w_pressure(pressure=0):
    def roulette_wheel(population, minimization, random_state, pressure=1):
        total = determine_the_total_fitness_seletion_phase(population, pressure)
        selection_population = copy.deepcopy(population.tolist())
        individual1 = determine_the_parent(selection_population, total)
        individual2 = determine_the_parent(selection_population, total)
        return individual1, individual2
    return roulette_wheel



def parametrize_roulette_wheel_pressure(pressure):
    def roulette_wheel(population, random_state, pressure=pressure):
        total = determine_the_total_fitness_seletion_phase(population, pressure)
        individual1 = determine_the_parent_pressure(population.tolist(), total, pressure, random_state)
        individual2 = determine_the_parent_pressure(population.tolist(), total, pressure, random_state)
        return individual1, individual2
    return roulette_wheel


#Need to confirm this. Somethimes the sames individuals are selected and it is annoying.
def parametrized_tournament_selection(pressure2 = 1):
    def tournament_selection(population, random_state, pressure):
        tournament_pool_size = int(len(population)*pressure)
        tournament_pool = random_state.choice(population, size=tournament_pool_size, replace=False)
        i = 2
        individual1 = tournament_pool[0]
        individual2 = tournament_pool[1]
        while i<tournament_pool_size:
            if tournament_pool[i].fitness**pressure > individual1.fitness**pressure:
                individual1 = tournament_pool[i]
            else:
                if tournament_pool[i].fitness**pressure>individual2.fitness**pressure:
                    individual2 = tournament_pool[i]
            i=i+1
        return individual1, individual2
    return tournament_selection


def parametrize_rank_selection (pressure=0):
    def rank_selection(population, random_state, pressure=0):
        rank_population = order_numpy_solutions_array(population, pressure)
        population_probabilities = calculate_probabilities_for_rank(len(population))
        individual1 = determine_the_selection_offspring(rank_population, population_probabilities)
        rank_population = order_numpy_solutions_array(rank_population)
        population_probabilities = calculate_probabilities_for_rank(len(rank_population))
        individual2 = determine_the_selection_offspring(rank_population, population_probabilities)
        return individual1, individual2
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


def order_numpy_solutions_array(population, pressure = 0):

    selectTheBestPopulation = pd.DataFrame(np.asanyarray([[individual, individual.fitness**pressure] for individual
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



def parametrize_botzmann_selection(pressure):
    def botzmann_selection(population, random_state, pressure):
        mean = calculate_media_solution(population)
        solution1 = select_the_solution_bolzmann(population, mean, random_state)
        solution2 = select_the_solution_bolzmann(population, mean, random_state)
        return solution1, solution2
    return botzmann_selection



def select_the_solution_bolzmann(population, mean, random_state):
    max_bound, mean_temperature = calculate_botzmann_total(population, mean)
    random_number = random_state.uniform(0, max_bound)
    solution1 = None
    probability = 0
    for i in range(len(population)):
        probability = math.exp((population[i].fitness / mean) / mean_temperature) + probability
        if random_number < probability:
            solution1 = population[i]
            break
    return solution1


def calculate_botzmann_total(population, mean):
    total = 0
    for i in range(len(population)):
        total = (math.exp(population[i].fitness/mean)) + total
    new_total = 0
    for i in range(len(population)):
        new_total = math.exp(population[i].fitness/mean)/math.exp(total/len(population)) + new_total
    return new_total, total/len(population)


def calculate_probabilities_for_rank(length):
    population_probabilities = []
    gauss_formula = (length + 1) * (length / 2)
    for i in range(length):
        population_probabilities.append(i / gauss_formula)
    return population_probabilities

def determine_the_selection_offspring(population, population_probabilities):
    sum = 0
    for j in range(len(population_probabilities)):
        if random.random() < population_probabilities[j] + sum:
            individual = population[j]
            del population[j]
            return individual
        sum = sum + population_probabilities[j]
    return population[:-1]

def determine_the_total_fitness_seletion_phase(population, pressure):
    sum = 0
    for i in range(len(population)):
        sum = sum + population[i].fitness**pressure
    return sum

def determine_the_total_fitness_seletion_phase_pressure(population, pressure):
    sum = 0
    for i in range(len(population)):
        sum = (sum + population[i].fitness)*pressure
    return sum

def determine_the_parent(probability_individuals_population, total):
    sum = 0
    i = 0
    while i < len(probability_individuals_population):
        if random.random() < (probability_individuals_population[i].fitness+sum)/total:
            individual = probability_individuals_population[i]
            del probability_individuals_population[i]
            return individual
        sum = sum + probability_individuals_population[i].fitness
        i=i+1
    return probability_individuals_population[:-1]

def determine_the_parent_pressure(probability_individuals_population, total, pressure, random_state):
    sum = 0
    i = 0
    while i < len(probability_individuals_population):
        if random_state.uniform() < ((probability_individuals_population[i].fitness**pressure)+sum)/total:
            individual = probability_individuals_population[i]
            del probability_individuals_population[i]
            return individual
        sum = sum + probability_individuals_population[i].fitness**pressure
        i = i+1

def calculate_feature_scaling(min, max, solution_position):
    return ((solution_position - min)/(max-min))


def parametrize_roulette_wheel_Wpressure_sharing_fitness(pressure):
    def roulette_wheel (population):
        selection_population = population.copy(deep=True)
        individuals = []
        i = 0
        while i < 2:
            random_number = random.random()
            selection_population = selection_population.loc[selection_population['probability'] < random_number]
            selection_population.sort_values(by=['probability'], inplace=True, ascending=False)
            print(selection_population.iloc[i]['Id'])
            i = i+1
        return individuals[0], individuals[1]
    return roulette_wheel


def normalize_distance(distance):
    return (1 / (1 - math.exp(-distance))) * 2 - 1


def distance_scheme(individual1, individual2):
    distance_map = []
    for i in range(len(individual1.representation)):
        distance_map.append((individual1.representation[i] - individual2.representation[i])**2)
    distance = 0
    for i in range(len(distance_map)):
        distance = distance + distance_map[i]
    return distance


def determine_the_total_fitness_seletion_phase_sharing_fitness(population1, pressure, iteration):
    candidates_list = []
    fitness_list = []
    raw_fitness = []
    total_fitness = 0
    distance = []
    raw_distance = []
    for i in range(len(population1)):
        fitness = population1[i].fitness
        distances = calculate_share_distance(population1[i], population1, pressure)
        share_distance = distances[0]
        raw_distance.append(distances[1])
        print('FITNESS: ',fitness)
        print('SHARE DISTANCE: ', share_distance)
        total_fitness = total_fitness + (fitness/share_distance)
        candidates_list.append(population1[i])
        fitness_list.append((fitness/share_distance))
        raw_fitness.append(fitness)
        distance.append(share_distance)
    candidates = pd.DataFrame(candidates_list, columns=['Id'])
    candidates['Fitness_Sharing'] = fitness_list
    candidates['Fitness_Sharing'] = pd.to_numeric(candidates['Fitness_Sharing'])
    candidates_tolist = candidates['Fitness_Sharing'].tolist()
    sum = 0
    for i in range(len(candidates_tolist)):
        sum =  sum + candidates_tolist[i]
    candidates['probability'] = candidates['Fitness_Sharing']/sum
    candidates['total_fitness'] = total_fitness
    candidates['Raw_Fitness'] = raw_fitness
    candidates['Distance'] = raw_distance
    plot = sbn.scatterplot(x='probability', y='Raw_Fitness', data=candidates)
    fig = plot.get_figure()
    fig.savefig("C:\\Users\\hppor\\Desktop\\Plots\\"+str(iteration))
    plt.clf()
    return candidates


def calculate_share_distance(individual, population, pressure):
    total_share_distance = 0
    total_raw = 0
    for i in range(len(population)):
        if population[i].id != individual.id:
            distance = distance_scheme(individual, population[i])
            print('Distance: ', distance)
            distance_normalize = normalize_distance(distance)
            print("Distance Normalize", distance_normalize)
            if distance_normalize < pressure:
                total_share_distance = total_share_distance + (1-(distance_normalize/pressure))
                total_raw = total_raw + distance
    return total_share_distance, total_raw


def media_crossover_point(p1_r, p2_r, random_search):
    off1, off2 = p1_r, p2_r

    for i in range(len(p1_r)):
        off1[i] = (p1_r[i]+p2_r[i])/2
    off2 = copy.deepcopy(off1)

    return off2, off1


def cicle_crossover(p1_r, p2_r, random_search):
    index1 = []
    for i in range(len(p1_r)):
        index1.append(i)

    index2 = copy.deepcopy(index1)
    shuffle(index1)
    shuffle(index2)

    another_list = []
    off1 = copy.deepcopy(p2_r)
    off2 = copy.deepcopy(p1_r)
    i = 0
    while i < len(off1):
        off1[i] = p1_r[i]
        i = index2[i]
        another_list.append(i)
        for j in range(len(another_list)):
            if another_list[j] is not None and another_list[j] == i:
                i = len(off1)
                break
        i = i+1
    another_list = []
    i = 0
    while i < len(off2):
        off2[i] = p2_r[i]
        i = index1[i]
        another_list.append(i)
        for j in range(len(another_list)):
            if another_list[j] is not None and another_list[j] == i:
                i = len(off1)
                break
        i = i+1
    return off1, off2

def cicle_crossover2(p1_r, p2_r, random_search):
    index1 = []
    for i in range(len(p1_r)):
        index1.append(i)

    index2 = copy.deepcopy(index1)
    shuffle(index1)
    shuffle(index2)

    another_list = []
    off1 = copy.deepcopy(p2_r)
    off2 = copy.deepcopy(p1_r)
    i = 0
    while i < len(off1):
        off1[i] = p1_r[i]
        i = index2[i]
        another_list.append(i)
        for j in range(len(another_list)):
            if another_list[j] is not None and another_list[j] == i:
                i = len(off1)
                break
        i = i+1
    another_list = []
    i = 0
    while i < len(off2):
        off2[i] = p2_r[i]
        i = index1[i]
        another_list.append(i)
        for j in range(len(another_list)):
            if another_list[j] is not None and another_list[j] == i:
                i = len(off1)
                break
        i = i+1

    pandas_to_sort_1 = pd.DataFrame(np.asanyarray(index1))
    pandas_to_sort_1.rename(index=str, columns={0: "Index"}, inplace=True)
    pandas_to_sort_1['Off1'] = off1
    pandas_to_sort_1.sort_values(ascending=False, inplace=True, by="Index")
    pandas_to_sort_1_list = pandas_to_sort_1['Off1'].tolist()

    pandas_to_sort_2 = pd.DataFrame(np.asanyarray(index2))
    pandas_to_sort_2.rename(index = str, columns={0: "Index"}, inplace=True)
    pandas_to_sort_2['Off2'] = off2
    pandas_to_sort_2.sort_values(ascending=False, inplace=True, by="Index")
    pandas_to_sort_2_list = pandas_to_sort_2['Off2'].tolist()




    return pandas_to_sort_1_list, pandas_to_sort_2_list


