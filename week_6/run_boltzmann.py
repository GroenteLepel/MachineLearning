#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:15:36 2019

@author: Daniel Kok, Laurens Sluyterman en Ludo van Alst
Advanced Machine Learning @ Radboud University
"""

# %% Import modules & defining constants
import copy
import matplotlib.pyplot as plt
import sys

from ising_ensemble import IsingEnsemble
sys.path.append("../week_3")
from ising_model import IsingModel
sys.path.append("../week_6")
import boltzmann


# %% Ising
n_spins = 10
n_models = 10
n_MC_state_samples = 500
ie = IsingEnsemble(n_models, n_spins)
ie_mc = copy.deepcopy(ie)
eta_start = 0.1

print("exact")
coupling_matrix_opt_exact, threshold_vector_opt_exact, llh_exact, iterations_exact = boltzmann.optimise(ie, eta=eta_start, method='exact', output=True)
print("mc")
coupling_matrix_opt_mc, threshold_vector_opt_mc, llh_mc, iterations_MC = boltzmann.optimise(ie_mc, eta=eta_start, method='mc', output=True, n_MC_state_samples = n_MC_state_samples, iterations_bound = iterations_exact)
plt.plot(llh_exact, label='exact')
plt.plot(llh_mc, label='mc')
plt.title('Boltzmann machine exact vs monte carlo\n{0:d} spins; {1:d} models'.format(n_spins, n_models))
plt.xlabel('iterations')
plt.ylabel('log-likelihood')
plt.legend()
plt.savefig('boltzmann_ising_s{0:d}m{1:d}.jpg'.format(n_spins, n_models), dpi=1000)
plt.show()

#%% Salamander
n_spins = 160
n_models = 283041
n_MC_state_samples = 10
n_salamander_data = 160
eta_start = 0.1
filepath_expectations = '../data/salamander_retina_expectations.txt'
filepath_retina = '../data/salamander_retina.txt'

expectation_matrix_c_salamander, expectation_vector_c_salamander = boltzmann.read_salamander_expectations(filepath_expectations, n_spins)

coupling_matrix_MF, threshold_vector_MF = boltzmann.mean_field_estimate(expectation_matrix_c_salamander, expectation_vector_c_salamander)

ie_n_salamander = boltzmann.n_salamander_states_to_ie(filepath_retina, n_salamander_data)
ie_n_salamander.expectation_matrix_c = copy.deepcopy(expectation_matrix_c_salamander)
ie_n_salamander.expectation_vector_c = copy.deepcopy(expectation_vector_c_salamander)

coupling_matrix_MC, threshold_vector_MC, iterations_MC_salamander = boltzmann.optimise(ie_n_salamander, eta=eta_start, method='mc', n_MC_state_samples = n_MC_state_samples, output=True)