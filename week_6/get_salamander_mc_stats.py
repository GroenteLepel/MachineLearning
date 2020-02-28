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

from ising_ensemble import IsingEnsemble
sys.path.append("../week_3")
from ising_model import IsingModel
sys.path.append("../week_6")
import boltzmann


#%% Declaring constants
n_spins = 160
n_models = 283041
n_MC_state_samples = 2000
n_salamander_data = 500
max_iterations = 50
eta_start = 0.1
burn_in_iterations = 500
output = True
filepath_expectations = '../data/salamander_retina_expectations.txt'
filepath_retina = '../data/salamander_retina.txt'

expectation_matrix_c_salamander, expectation_vector_c_salamander = boltzmann.read_salamander_expectations(filepath_expectations, n_spins)


#%% Functions
def get_salamander_mc_stats(n_salamander_data, max_iterations, output):
    ie_n_salamander = boltzmann.n_salamander_states_to_ie(filepath_retina, n_salamander_data)
    ie_n_salamander.expectation_matrix_c = copy.deepcopy(expectation_matrix_c_salamander)
    ie_n_salamander.expectation_vector_c = copy.deepcopy(expectation_vector_c_salamander)
    
    coupling_matrix_MC, threshold_vector_MC, llh_salamander, iterations_MC_salamander = \
        boltzmann.optimise(ie_n_salamander, 
                           eta=eta_start, 
                           method='mc', 
                           n_MC_state_samples = n_MC_state_samples, 
                           output=output, 
                           iterations_bound = max_iterations)
    
    pickle.dump(coupling_matrix_MC, open('coupling_matrix_MC_{0}it_{1}SD_{2}MC.p'.format(max_iterations,n_salamander_data,n_MC_state_samples),'wb'))
    pickle.dump(threshold_vector_MC, open('threshold_vector_MC_{0}it_{1}SD_{2}MC.p'.format(max_iterations, n_salamander_data,n_MC_state_samples),'wb'))
    


#%% Salamander
# Note that LLH in monte carlo salamander cannot be calculated since one would need to store all 283041 ising models.
# The LLH is a sum over all models..
get_salamander_mc_stats(n_salamander_data=n_salamander_data, max_iterations=max_iterations, output=output)
