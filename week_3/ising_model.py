import copy
import itertools
from itertools import combinations
import numpy as np


class IsingModel:

    def __init__(self, n: int, frustrated: bool, threshold: bool):
        self.n = n
        global GLOBAL_STATES
        GLOBAL_STATES = np.array(list(map(list, itertools.product([-1, 1], repeat=self.n))))

        self.frustrated = frustrated
        self.threshold = threshold

        self.coupling_matrix = self._generate_matrix()
        self.threshold_vector = self._generate_thresholds()
        self.state = self._generate_spin_state()

        self.normalisation_constant = 0

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

    def _generate_thresholds(self):
        # Generate a threshold vector for the Boltzmann-Gibbs distribution
        # TODO: elaborate why we choose the normal distribution.
        if self.threshold:
            return np.random.normal(size=(self.n,))
        else:
            return np.zeros((self.n,))

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
        return -0.5 * np.dot(self.state,
                             np.dot(self.coupling_matrix, self.state)) \
               - np.dot(self.threshold_vector, self.state)

    def _find_normalisation_constant(self):
        # Find normalisation constant Z=sum(-E(s)), sum over all states s
        dummy = IsingModel(self.n,
                           frustrated=self.frustrated,
                           threshold=self.threshold)

        normalisation_constant = 0
        for state in GLOBAL_STATES:
            dummy.state = state
            normalisation_constant += np.exp(- dummy.ising_energy())

        return normalisation_constant

    def update_normalisation_constant(self):
        self.normalisation_constant = self._find_normalisation_constant()

    def p(self):
        """
        Boltzmann-Gauss distribution for this Ising Model with variables
        of all the spin states and its coupling matrix. Computes the
        normalisation constant first, and uses the Ising energy from this model.
        :return:  1D-float.
        """
        if self.normalisation_constant == 0:
            self.update_normalisation_constant()

        return 1. / self.normalisation_constant * np.exp(-self.ising_energy())

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
        :return: float, estimated maximum energy difference.
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

        return max_e_diff
