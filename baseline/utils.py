from functools import reduce
import numpy as np

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


def sphere_function(point):
    return np.sum(np.power(point, 2.), axis=len(point.shape) % 2 - 1)


def one_point_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    point = random_state.randint(len_)
    off1_r = np.concatenate((p1_r[0:point], p2_r[point:len_]))
    off2_r = np.concatenate((p2_r[0:point], p1_r[point:len_]))
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