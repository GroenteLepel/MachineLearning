import random
from itertools import combinations

import numpy as np

"""
Combinatoric optimisation
Theory
- Ising model (brief)
- Explain both II and SA
  - Psuedo-code in appendix
  - how are II and SA related?
  - What are the advantages of each method? (or what do you expect)
"""


class IsingOptimiser:
    np.random.seed(4)

    def __init__(self, n=100, frustrated=True):
        """
        Initialise the class and all its parameters at once. These can be used
        throughout the class by calling them via self.x, with x the parameter
        which you want to use.
        :param n: amount of points to generate
        :param frustrated: indicates if we are dealing with a frustrated system
        or not (i.e. if w is only > 0 or if there are values < 0 as well.
        """
        self.n = n
        self.frustrated = frustrated
        self.flip_reg = np.zeros(self.n, dtype=bool)

        self.weights = self._generate_matrix()
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

    def ising_energy(self):
        return -0.5 * np.dot(self.state, np.dot(self.weights, self.state))

    def iterative_improvement(self, neighbourhood=3):
        old_cost = self.ising_energy()
        indices = np.arange(self.n)

        for n_flips in range(1, neighbourhood + 1):
            np.random.shuffle(indices)
            flip_combs_dummy = list(combinations(indices, n_flips))
            flip_combs = [np.array(x) for x in flip_combs_dummy]

            for flip_comb in flip_combs:
                self.flip_state(flip_comb)
                if self.ising_energy() < old_cost:
                    return True
                self.flip_state(flip_comb)

        return False
