import numpy as np
from ising_model import IsingModel

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

    def __init__(self, ising_model: IsingModel):
        """
        Initialise the class and all its parameters at once. These can be used
        throughout the class by calling them via self.x, with x the parameter
        which you want to use.
        :param n: amount of points to generate
        :param frustrated: indicates if we are dealing with a frustrated system
        or not (i.e. if w is only > 0 or if there are values < 0 as well.
        """
        self.im = ising_model

    def iterative_improvement(self, neighbourhood=3):
        old_cost = self.im.ising_energy()

        for n_flips in range(1, neighbourhood + 1):
            flip_combinations = self.im.generate_flip_combinations(n_flips)

            for flip_comb in flip_combinations:
                self.im.flip_state(flip_comb)
                if self.im.ising_energy() < old_cost:
                    return True
                self.im.flip_state(flip_comb)

        return False

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

        # Estimate the maximum dE for flipping into a certain neighbourhood
        max_de = self.im.estimate_max_energy_diff(neighbourhood)
        beta_init = 1 / max_de  # sets initial temperature

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
                current_energy = self.im.ising_energy()
                self.im.flip_state(comb)
                candidate_energy = self.im.ising_energy()

                # Perform metropolis hastings step
                if not self.accept(beta, current_energy, candidate_energy):
                    self.im.flip_state(comb)

                energy_array[t1] = self.im.ising_energy()

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
