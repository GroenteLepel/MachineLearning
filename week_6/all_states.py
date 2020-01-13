import itertools
import numpy as np


def gen_all_possible_states(n_spins):
    lst = list(map(list, itertools.product([-1, 1], repeat=n_spins)))
    return np.array(lst)