import random
from itertools import permutations

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

    # def check_Emin(self, states_to_flip, indices_to_flip):
    #     indices_not_flipped = [x for x in np.linspace(self.n) if x not in indices_to_flip]
    #     for index in indices_not_flipped:
    #         indices_to_flip += [index]
    #         check_Emin(self, states_to_flip-1, indices_to_flip)
    #
    #         old_energy = self.ising_energy()
    #         for i in indices_to_flip:
    #             self.flip_state(i)
    #             if old_energy > self.ising_energy():
    #                 # Lower energy achieved, end search.
    #
    #                 return True
    #             # Flip back to the previous state where the energy was better.
    #             self.flip_state(i)
    #
    # def find_neighbourhood(self, state, size):
    #     for i in range(1, size + 1):
    #         new_state = check_Emin(self.state, i)

    def iterative_improvement(self, neighbourhood=3):
        old_cost = self.ising_energy()
        indices = np.arange(self.n)

        if neighbourhood > 1:
            returni self.iterative_improvement(neighbourhood=neighbourhood-1)

        if neighbourhood > 1:
            perms_iter = permutations(indices, neighbourhood)
            perms = [np.array(x) for x in list(perms_iter)]
        else:
            perms = indices

        random.shuffle(perms)

        for p in perms:
            # p is of class tuple. In order to do things with it, we need to
            #  make it an array.
            p_arr = np.array(p)
            self.flip_state(p_arr)
            if self.ising_energy() < old_cost:
                # self.flip_reg[p_arr] = True
                return True
            else:
                # This state does not make it better, reverse the flip.
                self.flip_state(p_arr)

        return False




