import numpy as np


def sum_values(items):
  return np.sum([item.value for item in items])


def sum_weights(items):
  return np.sum([item.weight for item in items])


def random_items(n_items, max_weight, max_value, random_state):
  return np.array([_Item(weight=random_state.uniform(high=max_weight),
               value=random_state.uniform(high=max_value)) for _ in range(n_items)])


class _Item:


    _id = 0


    def __init__(self, weight, value):
        self._item_id = _Item._id
        _Item._id += 1
        self.weight = weight
        self.value = value


    def __str__(self):
        return "Item#%d - weigth: %.2f value: %.2f" % (self._item_id, self.weight, self.value)


    def __repr__(self):
        return self.__str__()


    @property
    def value(self):
        return self.__value


    @value.setter
    def value(self, value):
        if value < 0:
            self.__value = 0.01
        elif value > 100:
            self.__value = 100
        else:
            self.__value = value


    @property
    def weight(self):
        return self.__weight


    @weight.setter
    def weight(self, weight):
        if weight < 0.01:
            self.__weight = 0.01
        elif weight > 99:
            self.__weight = 99
        else:
            self.__weight = weight