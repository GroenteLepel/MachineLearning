import copy
from itertools import combinations

import numpy as np


class IsingModel:

    def __init__(self, n: int, frustrated: bool):
        self.n = n
        self.frustrated = frustrated

        self.coupling_matrix = self._generate_matrix()
        self.state = self._generate_spin_state()

    def _generate_matrix(self):
        # Generate a nxn matrix, we chose normal distribution
        # TODO: explain why we chose normal distribution.
        w = np.random.normal(size=(self.n, self.n))

        # Make it symmetric, and normalise
        w += w.transpose()
        w /= 2

        # Set diagonal to zero, no spin interactions with itself
        np.fill_diagonal(w, 0)

        if not self.frustrated:
            w[w < 0] *= -1

        return w

    def _generate_spin_state(self):
        return np.random.choice([-1, 1], size=(self.n,))

    def flip_state(self, i):
        self.state[i] *= -1

    def generate_flip_combinations(self, neighbourhood):
        indices = np.arange(self.n)
        np.random.shuffle(indices)
        flip_combs_iterator = list(combinations(indices, neighbourhood))
        flip_combinations = [np.array(x) for x in flip_combs_iterator]

        return flip_combinations

    def ising_energy(self):
        return -0.5 * np.dot(self.state, np.dot(self.coupling_matrix, self.state))

    def estimate_max_energy_diff(self, neighbourhood):
        """
        Estimate the max energy difference of the ising model system based on
        the amount of spins that are flipped.

        It is called "estimate", so we figured how we would do it is as
        follows:
        - generate all possible combinations of spin flips in the max
         neighbourhood, since this has the highest probability of containing
         the largest energy difference (?)
        - calculate the first i energies of these combinations
        - pick the max difference

        :param neighbourhood:
        :return:
        """
        assert neighbourhood <= self.n, "Value of neighbourhood exceeds the amount of spins in the state."

        max_e_diff = 0
        initial_energy = self.ising_energy()
        neighbour = copy.deepcopy(self)

        for i in range(int(self.n / 2)):
            combination = np.random.choice(self.n, neighbourhood, replace=False)
            neighbour.flip_state(combination)
            new_energy = neighbour.ising_energy()
            new_e_diff = np.abs(new_energy - initial_energy)
            if new_e_diff > max_e_diff:
                max_e_diff = new_e_diff

        print("Estimated max e diff:")
        print(max_e_diff)
        return max_e_diff
