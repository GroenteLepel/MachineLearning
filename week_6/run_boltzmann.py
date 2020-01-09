#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:15:36 2019

@author: Daniel Kok, Laurens Sluyterman en Ludo van Alst
Advanced Machine Learning @ Radboud University
"""

# %% Import modules & defining constants
from week_3.ising_model import IsingModel
from week_6 import boltzmann
from week_6.ising_ensemble import IsingEnsemble
import matplotlib.pyplot as plt

n_spins = 10
n_models = 10

# %% Main
ie = IsingEnsemble(n_models, n_spins)

llh = boltzmann.boltzmann_optimiser(ie)
plt.plot(llh)
plt.show()