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

n_spins = 160
n_models = 10

# %% Main
#ie = IsingEnsemble(n_models, n_spins)
#ie_mc = copy.deepcopy(ie)
#
#print("exact")
#llh_exact = boltzmann.boltzmann_optimiser(ie, method='exact', output=True)
#print("mc")
#llh_mc = boltzmann.boltzmann_optimiser(ie_mc, method='mc',  output=True)
#plt.plot(llh_exact, label='exact')
#plt.plot(llh_mc, label='mc')
#plt.legend()
#plt.show()

coupling_matrix_MF, threshold_vector_MF = boltzmann.mean_field_estimate('../data/salamander_retina_expectations.txt', 160)
print(coupling_matrix_MF)
print()
print(threshold_vector_MF)