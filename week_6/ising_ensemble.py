import numpy as np
import copy
import itertools
import sys

sys.path.append("../week_3")
from ising_model import IsingModel
sys.path.append("../week_6")
import all_states as all_states

class IsingEnsemble:

    def __init__(self, n_models, n_spins, frustrated=True):

        self.n_models = n_models
        self.n_spins = n_spins
        self.frustrated = frustrated

        self.state_set = [IsingModel(n_spins, frustrated, threshold=True) for _
                          in range(n_models)]
        self.coupling_matrix = copy.deepcopy(self.state_set[0].coupling_matrix)
        self.threshold_vector = copy.deepcopy(self.state_set[0].threshold_vector)

        # TODO: this is different for each Ising Model, so not the same for
        #  all like we do now.
        if n_spins <= 10:
            self.normalisation_constant = self._find_normalisation_constant()
        else:
            self.normalisation_constant = 0

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
            # This is not a reference. This is just to make sure that in the
            #  initial state, all normalisation constants are equal.
            state.normalisation_constant = self.normalisation_constant

    def _find_normalisation_constant(self):
        # Find normalisation constant Z=sum(-E(s)), sum over all states s
        dummy = IsingModel(self.n_spins,
                           frustrated=self.frustrated,
                           threshold=True)
        dummy.coupling_matrix = self.coupling_matrix
        dummy.threshold_vector = self.threshold_vector

        states = all_states.gen_all_possible_states(self.n_spins)
        normalisation_constant = 0
        for state in states:
            dummy.state = state
            normalisation_constant += np.exp(- dummy.ising_energy())

        return normalisation_constant

    def update_normalisation_constants(self):
        self.normalisation_constant = self._find_normalisation_constant()
        for model in self.state_set:
            model.set_normalisation_constant(self.normalisation_constant)

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
        expectation = expectation / self.n_models

        return expectation

    def _clamped_expectation_matrix(self):
        expectation = np.zeros(shape=(self.n_spins, self.n_spins))
        for state in self.state_set:
            expectation += np.outer(state.state, state.state)
        expectation = expectation / self.n_models

        return expectation
