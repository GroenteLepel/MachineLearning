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

    def generate_flip_combinations(self, neighbourhood):
        indices = np.arange(self.n)
        np.random.shuffle(indices)
        flip_combs_iterator = list(combinations(indices, neighbourhood))
        flip_combinations = [np.array(x) for x in flip_combs_iterator]

        return flip_combinations

    def ising_energy(self):
        return -0.5 * np.dot(self.state, np.dot(self.weights, self.state))

    def iterative_improvement(self, neighbourhood=3):
        old_cost = self.ising_energy()

        for n_flips in range(1, neighbourhood + 1):
            flip_combinations = self.generate_flip_combinations(n_flips)

            for flip_comb in flip_combinations:
                self.flip_state(flip_comb)
                if self.ising_energy() < old_cost:
                    return True
                self.flip_state(flip_comb)

        return False

    def energy_distribution(self, beta):
        """
        probability density (not normalised) to sample from using metropolis
        hastings.
        :param beta:
        :return:
        """
        return np.exp(-beta * self.ising_energy())

    # def metropolis_hastings(self, beta, n_points):
    #     samples = np.zeros(n_points)
    #
    #     index = 0
    #     while samples[-1] == 0:
    #         new_sample = self._generate_spin_state()
    #
    #         difference = p_simulated_annealing()
    #
    #         index += 1

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
        max_e_diff = 0
        initial_energy = self.ising_energy()
        for i in range(100):
            combination = np.random.choice(self.n, neighbourhood, replace=False)
            self.flip_state(combination)
            new_energy = self.ising_energy()
            new_e_diff = np.abs(new_energy - initial_energy)
            if max_e_diff < new_e_diff:
                max_e_diff = new_e_diff
            initial_energy = new_energy

        print("Estimated max e diff:")
        print(max_e_diff)
        return max_e_diff

    def simulated_annealing(self, neighbourhood,
                            length_markov_chain=2000, n_betas=1000):
        """
        Copy-paste of simulated annealing method delivered to us in the
        exercise. By no means is this a good bit of code. Improve.
        TODO:
          [x] rename parameters so it is actually readable
          [x] implement the estimate_max_energy_diff code (and come up with a
              better name for this function)
          [ ] optimise the while loop so increments do not take place inside the
              loop (enumerate, itertools?)
          [x] run and test.
        :param neighbourhood:
        :param length_markov_chain:
        :param n_betas:
        :return:
        """
        # beta_list = 1.01 * np.logspace(1, n_betas, base=beta_init)

        mean_energies = np.zeros(n_betas)  # Stores the mean energy at each beta
        stdev_energies = np.zeros(n_betas)  # Stores std energy at each beta

        # Estimate the maximum dE for a certain spin flip
        # max_de = self.estimate_max_energy_diff(neighbourhood)
        # beta_init = 1 / max_de  # sets initial temperature
        beta_init = 0.0091

        factor = 1.01  # increment of beta at each new chain

        t_count = 0
        stdev_energies[t_count] = 1
        beta = beta_init
        while stdev_energies[t_count] > 0:
            # increment
            t_count += 1
            beta *= factor
            # Initialise energy_array
            energy_array = np.zeros(length_markov_chain)

            for t1 in range(length_markov_chain):
                comb = np.random.choice(self.n, neighbourhood, replace=False)

                # Choose new x by flipping to new neighbourhood
                current_energy = self.ising_energy()
                self.flip_state(comb)
                candidate_energy = self.ising_energy()

                # Perform metropolis hastings step
                if not self.accept(beta, current_energy, candidate_energy):
                    self.flip_state(comb)

                energy_array[t1] = self.ising_energy()

            mean_energies[t_count] = np.mean(energy_array)
            stdev_energies[t_count] = np.std(energy_array)
            # print(t_count, beta,
            #       mean_energies[t_count], stdev_energies[t_count])

        return t_count, mean_energies, stdev_energies

    def accept(self, beta, current, candidate):
        difference = candidate - current
        a = np.exp(- beta * difference)

        if a >= 1:
            return True
        elif np.random.rand() <= a:
            return True
        else:
            return False
