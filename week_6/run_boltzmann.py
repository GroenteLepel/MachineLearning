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
import pickle
import numpy as np

from get_observed_rates import get_observed_rates
from ising_ensemble import IsingEnsemble
sys.path.append("../week_3")
from ising_model import IsingModel
sys.path.append("../week_6")
import boltzmann


# %% Ising exact vs MC
# n_spins = 10
# n_models = 10
# n_MC_state_samples = 500
# ie = IsingEnsemble(n_models, n_spins)
# ie_mc = copy.deepcopy(ie)
# ie_mf = copy.deepcopy(ie)
# eta_start = 0.1
# output = True
#
# coupling_matrix_opt_exact, threshold_vector_opt_exact, llh_exact, iterations_exact = boltzmann.optimise(ie, eta=eta_start, method='exact', output=output)
# coupling_matrix_opt_mc, threshold_vector_opt_mc, llh_mc, iterations_MC = boltzmann.optimise(ie_mc, eta=eta_start, method='mc', output=output, n_MC_state_samples = n_MC_state_samples, iterations_bound = iterations_exact)
# coupling_matrix_opt_MF, threshold_vector_opt_MF = boltzmann.mean_field_estimate(ie_mf.expectation_matrix_c, ie_mf.expectation_vector_c)
# plt.plot(llh_exact, label='exact')
# plt.plot(llh_mc, label='mc')
# plt.title('Boltzmann machine exact vs monte carlo\n{0:d} spins; {1:d} models'.format(n_spins, n_models))
# plt.xlabel('iterations')
# plt.ylabel('log-likelihood')
# plt.legend()
# plt.savefig('boltzmann_ising_s{0:d}m{1:d}.jpg'.format(n_spins, n_models), dpi=1000)
# plt.show()


#%% Ising MC for different amount of samples
# sample_sizes = [100,200,500,700,800,1000,2000]
# n_spins = 10
# n_models = 10
# ie = IsingEnsemble(n_models, n_spins)
# ie_storage = copy.deepcopy(ie)
# eta_start = 0.1
# output = False
# fixed_seed = True
#
# average_iteration_list = [0,0,0,0,0,0,0]
# for i, n_MC_state_samples in enumerate(sample_sizes):
#     iteration_sum = 0.0
#     for _ in range(5):
#         coupling_matrix_opt_mc, threshold_vector_opt_mc, llh_mc, iterations_MC = \
#             boltzmann.optimise(ie, eta=eta_start, method='mc', output=output, n_MC_state_samples = n_MC_state_samples)
#         iteration_sum += iterations_MC
#         ie = copy.deepcopy(ie_storage)
#     average_iteration_list[i] = float(iteration_sum) / 5
#     print(sample_sizes[i], average_iteration_list[i])
    

#%% Extra ising sample sizes
# extra_sizes = [150,300,400,600]
# n_spins = 10
# n_models = 10
# eta_start = 0.1
# output = False
# fixed_seed = True
#
# extra_average_iteration_list = [0,0,0,0]
# for i, n_MC_state_samples in enumerate(extra_sizes):
#     iteration_sum = 0.0
#     for _ in range(5):
#         coupling_matrix_opt_mc, threshold_vector_opt_mc, llh_mc, iterations_MC = \
#             boltzmann.optimise(ie, eta=eta_start, method='mc', output=output, n_MC_state_samples = n_MC_state_samples)
#         iteration_sum += iterations_MC
#         ie = copy.deepcopy(ie_storage)
#     extra_average_iteration_list[i] = float(iteration_sum) / 5
#     print(extra_sizes[i], extra_average_iteration_list[i])
    

#%% Plot the Ising MC sample size vs iterations
# Below values are gathered from the above data in seperate runs
# thus results are hardcoded here.
# sizes = [100,150,200,300,400,500,600,700,800,1000,2000]
# iterations = [154.4,109.0,95.4,93.0,80.8,75.8,76.0,77.6,72.6,71.6,69.8]
#
# plt.scatter(sizes,iterations)
# plt.xlabel('# MC samples')
# plt.ylabel('Iterations')
# plt.title('Iterations vs # MC samples')
# plt.savefig('MC_samples_vs_iterations.png', dpi=1000)
# plt.show()
    

#%% Salamander get optimisation stats
output = True
itb = 600
observed_rates = get_observed_rates('../data/salamander_retina.txt')

filepath_expectations = '../data/salamander_retina_expectations.txt'
filepath_retina = '../data/salamander_retina.txt'
expectation_matrix_c_salamander, expectation_vector_c_salamander = \
    boltzmann.read_salamander_expectations(filepath_expectations, 160)


ie_P2 = IsingEnsemble(1, 10)
ie_P2.expectation_matrix_c = expectation_matrix_c_salamander[0:10][0:10]
ie_P2.expectation_vector_c = expectation_vector_c_salamander[0:10]
coupling_matrix_exact_retina_P2, threshold_vector_exact_retina_P2, _, _ = \
            boltzmann.optimise(ie_P2, eta=0.1, method='exact', output=output, iterations_bound=itb)

ie_P1 = IsingEnsemble(1,10)
empty_coupling_matrix = np.zeros((10,10), dtype=float)
ie_P1.expectation_matrix_c = empty_coupling_matrix
ie_P1.coupling_matrix = empty_coupling_matrix
ie_P1._reference_coupling_matrix()
ie_P1.expectation_vector_c = expectation_vector_c_salamander[0:10]
_, threshold_vector_exact_retina_P1, _, _ = \
            boltzmann.optimise(ie_P1, eta=0.1, method='exact', output=output, iterations_bound=itb)

#%% Salamander plot
# coupling_submatrix = coupling_matrix_MC_retina[0:10,0:10]
# threshold_subvector = threshold_vector_MC_retina[0:10]
coupling_submatrix_P2 = coupling_matrix_exact_retina_P2
threshold_subvector_P2 = threshold_vector_exact_retina_P2

im_10_retina_P2 = IsingModel(10, frustrated=True,threshold=True)
im_10_retina_P2.coupling_matrix = copy.deepcopy(coupling_submatrix_P2)
im_10_retina_P2.threshold_vector = copy.deepcopy(threshold_subvector_P2)
im_10_retina_P2.update_normalisation_constant()

## Now for P1

threshold_subvector_P1 = threshold_vector_exact_retina_P1

empty_coupling_matrix = np.zeros((10,10), dtype=float)
im_10_retina_P1 = IsingModel(10, frustrated=True, threshold=True)
im_10_retina_P1.coupling_matrix = copy.deepcopy(empty_coupling_matrix)
im_10_retina_P1.threshold_vector = copy.deepcopy(threshold_subvector_P1)
im_10_retina_P1.update_normalisation_constant()


P1_approx = np.zeros(2**10, dtype=float)
P2_approx = np.zeros(2**10, dtype=float)

for state_int_repr in range(2**10):
    state_bin_repr = int(bin(state_int_repr)[2:])
    state_bin_repr_str = f'{state_bin_repr:010}'
    state = list(state_bin_repr_str)
    new_state = [-1.0 if state[i] == '0' else float(state[i]) for i in range(10)]
    np_state = np.array(new_state, dtype=float)
    im_10_retina_P1.state = np_state
    im_10_retina_P2.state = np_state

    P1_approx[state_int_repr] = im_10_retina_P1.p() * 50
    P2_approx[state_int_repr] = im_10_retina_P2.p() * 50


plt.scatter(observed_rates, P1_approx, c='grey', marker='.', label='P1')
plt.scatter(observed_rates, P2_approx, c='red', marker='.', label='P2')
plt.xlabel('Observed pattern rate ($s^{-1}$)')
plt.ylabel('Approximated pattern rate ($s^{-1}$)')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**-4, 10**2)
plt.ylim(10**-10, 10**2)
plt.plot(observed_rates, observed_rates, c='black')
plt.legend(loc='lower right')
plt.savefig('fig2a_homemade_newww.png', dpi=1000)
plt.show()


#%% Salamander
# Note that LLH in monte carlo salamander cannot be calculated since one would need to store all 283041 ising models.
# The LLH is a sum over all models..
# n_spins = 160
# n_models = 283041
# n_MC_state_samples = 500
# n_salamander_data = 100
# max_iterations = 15
# eta_start = 0.1
# burn_in_iterations = -1
# output = True
# filepath_expectations = '../data/salamander_retina_expectations.txt'
# filepath_retina = '../data/salamander_retina.txt'
#
# expectation_matrix_c_salamander, expectation_vector_c_salamander = boltzmann.read_salamander_expectations(filepath_expectations, n_spins)
#
# coupling_matrix_MF, threshold_vector_MF = boltzmann.mean_field_estimate(expectation_matrix_c_salamander, expectation_vector_c_salamander)
# coupling_matrix_MF -= np.identity(n_spins) * coupling_matrix_MF
#
# ie_n_salamander = boltzmann.n_salamander_states_to_ie(filepath_retina, n_salamander_data)
# ie_n_salamander.expectation_matrix_c = copy.deepcopy(expectation_matrix_c_salamander)
# ie_n_salamander.expectation_vector_c = copy.deepcopy(expectation_vector_c_salamander)
#
# coupling_matrix_MC_retina = pickle.load(open('coupling_matrix_MC_50it_500SD_2000MC.p', 'rb'))
# threshold_vector_MC_retina = pickle.load(open('threshold_vector_MC_50it_500SD_2000MC.p', 'rb'))
#
# absdiff = np.abs(coupling_matrix_MC_retina - coupling_matrix_MF)
# absdiff_list = absdiff.reshape((1,-1))
            
#plt.hist(absdiff_list, bins=10)