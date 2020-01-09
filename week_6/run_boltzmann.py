#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:15:36 2019

@author: Daniel Kok, Laurens Sluyterman en Ludo van Alst
Advanced Machine Learning @ Radboud University
"""

# %% Import modules & defining constants
import copy

from week_3.ising_model import IsingModel
from week_6 import boltzmann
from week_6.ising_ensemble import IsingEnsemble
import matplotlib.pyplot as plt

n_spins = 5
n_models = 10

# %% Main
ie = IsingEnsemble(n_models, n_spins)
ie_mc = copy.deepcopy(ie)

print("exact")
llh_exact = boltzmann.boltzmann_optimiser(ie, method='exact', output=True)
print("mc")
llh_mc = boltzmann.boltzmann_optimiser(ie_mc, method='mc',  output=True)
plt.plot(llh_exact, label='exact')
plt.plot(llh_mc, label='mc')
plt.legend()
plt.show()