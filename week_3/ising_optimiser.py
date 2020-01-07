import copy

import numpy as np
from week_3.ising_model import IsingModel
from week_2.metropolis_hastings import accept_exponent

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

    def __init__(self, ising_model: IsingModel, neighbourhood: int):
        """
        Initialise the class and all its parameters at once. These can be used
        throughout the class by calling them via self.x, with x the parameter
        which you want to use.
        :param n: amount of points to generate
        :param frustrated: indicates if we are dealing with a frustrated system
        or not (i.e. if w is only > 0 or if there are values < 0 as well.
        """
        self.im = ising_model
        self.init_im = copy.deepcopy(ising_model)
        self.neighbourhood = neighbourhood

    def reset(self):
        self.im = copy.deepcopy(self.init_im)

    def optimise(self, method: str):
        switcher = {
            "iter": lambda: self._optimise_iteratively(),
            "sa": lambda: self._simulated_annealing()
        }
        method = switcher.get(method, lambda: "No proper method given")
        return method()

    def _optimise_iteratively(self):
        # print("Optimising iteratively.")
        while self._iterative_improvement_found():
            pass

    def _iterative_improvement_found(self):
        """
        Check all possible combinations of spin flips in the given
        neighbourhood of the optimiser, and returns True if one of these
        combinations contains a lower ising energy than the previous model.
        Returns false if none of the combinations result in a lower energy (
        i.e. a minimum is found).

        :return: boolean
        """
        old_cost = self.im.ising_energy()

        for n_flips in range(1, self.neighbourhood + 1):
            flip_combinations = self.im.generate_flip_combinations(n_flips)

            for flip_comb in flip_combinations:
                self.im.flip_state(flip_comb)
                if self.im.ising_energy() < old_cost:
                    return True
                self.im.flip_state(flip_comb)

        return False

    def _simulated_annealing(self,
                             length_markov_chain=2000, n_betas=1000):
        """
        Copy-paste of simulated annealing method delivered to us in the
        exercise. By no means is this a good bit of code. Improve.

        TODO:
          [ ] optimise the while loop so increments do not take place inside the
              loop (enumerate, itertools?)

        :param length_markov_chain:
        :param n_betas:
        :return:
        """
        print("Optimising using simulated annealing.")

        mean_energies = np.zeros(n_betas)  # Stores the mean energy at each beta
        stdev_energies = np.zeros(n_betas)  # Stores std energy at each beta

        # Estimate the maximum dE for flipping into a certain neighbourhood
        factor = 1.01  # increment of beta at each new chain
        # max_de = self.im.estimate_max_energy_diff(self.neighbourhood)
        # beta_init = 1 / max_de  # sets initial temperature
        beta_init = 1 / 100
        beta_list = beta_init * np.logspace(1, n_betas,
                                            num=n_betas, base=factor)

        # Check energy of current state
        current_energy = self.im.ising_energy()
        t_count = 0
        stdev_energies[t_count] = 1
        # while stdev_energies > 440:
        while stdev_energies[t_count] > 1e-12:
            t_count += 1

            # Initialise energy_array
            tmp_energy_array = np.zeros(length_markov_chain)

            for i in range(length_markov_chain):
                # Generate a combination of size neighbourhood to flip the spins
                comb = np.random.choice(self.im.n, self.neighbourhood,
                                        replace=False)

                # Generate a candidate new state
                candidate_im = copy.deepcopy(self.im)
                candidate_im.flip_state(comb)
                candidate_energy = candidate_im.ising_energy()

                # Check if new candidate state is accepted according to
                #  metropolis hastings
                if accept_exponent(current_energy, candidate_energy,
                                   factor=beta_list[t_count]):
                    self.im = candidate_im
                    current_energy = candidate_energy

                tmp_energy_array[i] = current_energy

            mean_energies[t_count] = np.mean(tmp_energy_array)
            stdev_energies[t_count] = np.std(tmp_energy_array)
            if t_count % 100 == 0:
                print('{}: mean: {}, stdev: {}.'.format(t_count,
                                                        mean_energies[t_count],
                                                        stdev_energies[
                                                            t_count]))

        return t_count, beta_list[:t_count], \
               mean_energies[:t_count], stdev_energies[:t_count]