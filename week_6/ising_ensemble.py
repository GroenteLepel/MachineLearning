import numpy as np

from week_3.ising_model import IsingModel
import copy
import itertools

class IsingEnsemble:

    def __init__(self, n_models, n_spins, frustrated=True):

        self.n_models = n_models
        self.n_spins = n_spins
        self.frustrated = frustrated

        self.state_set = [IsingModel(n_spins, frustrated, threshold=True) for _
                          in range(n_models)]
        self.coupling_matrix = copy.deepcopy(self.state_set[0].coupling_matrix)
        self.threshold_vector = copy.deepcopy(self.state_set[0].threshold_vector)

        self.state_set[0].find_normalisation_constant()
        self.normalisation_constant = self.state_set[0].normalisation_constant

        self.expectation_vector_c = self._clamped_expectation_vector()
        self.expectation_matrix_c = self._clamped_expectation_matrix()

        # Reference all the coupling matrices of the state set to the main
        #  coupling matrix of this class.
        self._reference_coupling_matrix()

    def _reference_coupling_matrix(self):
        for state in self.state_set:
            # Reference all state vectors to one global, main coupling matrix or
            #  treshold vector. This means that if one of these global, main
            #  values change, all the states change along.
            state.coupling_matrix = self.coupling_matrix
            state.threshold_vector = self.threshold_vector
            state.normalisation_constant = self.normalisation_constant

    def LLH(self):
        # Calculate the log-likelihood for the set of states (par 2.5 of the reader)
        loglikelihood = 0
        for im in self.state_set:
            loglikelihood += np.log(im.p())
        return 1. / self.n_models * loglikelihood

    def _clamped_expectation_vector(self):
        expectation = np.zeros(self.n_spins)
        for state in self.state_set:
            expectation += state.state
        expectation /= self.n_models

        return expectation

    def _clamped_expectation_matrix(self):
        expectation = np.zeros(shape=(self.n_spins, self.n_spins))
        for state in self.state_set:
            expectation += np.outer(state.state, state.state)
        expectation /= self.n_models

        return expectation