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

def metropolis_hastings(n_points, n_dims=3):
    """
    Method for sampling data from a distribution p
    :param p: distribution to sample using this method
    :param n_points: amount of points you want to sample from distribution p
    :param n_dims: amount of dimensions of the points which are sampled
    :return: array with shape (n_points, n_dims)
    """

    samples = np.zeros((n_points, n_dims))
    x = np.zeros(n_dims)

    for i in range(n_points):
        if i % (n_points / 4) == 0:
            print(i)
        new_x = x + np.random.normal(size=n_dims, scale=0.1)

        counter = 0
        if accept_exponent(objective_function, data, labels, new_x, x):
            samples[i] = new_x
            x = new_x
        else:
            samples[i] = x

    return samples


def accept_exponent(f, data, labels, sample_x, current_state):
    # TODO: Remove this ugly specific implementation.
    difference = f(sample_x, data, labels, 0.01) - f(current_state, data,
                                                     labels, 0.01)
    a = np.exp(- difference)

    if a >= 1:
        return True
    elif np.random.rand() <= a:
        return True
    else:
        return False


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

    def p_simulated_annealing(self, beta):
        return np.exp(-beta * self.ising_energy())
    
    def metropolis_hastings(self, beta, n_points):
        samples = np.zeros(n_points)
        
        index = 0
        while samples[-1] == 0:
            new_sample = self._generate_spin_state()
            
            difference = p_simulated_annealing()
            
            index += 1
        

    def simulated_annealing(self, beta_init = 0.0091, markov_chain_length = 2000, n_betas = 450):
        beta_list = 1.01 * np.logspace(1, n_betas, base = beta_init)    