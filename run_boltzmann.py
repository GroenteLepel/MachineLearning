#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:15:36 2019

@author: Daniel Kok, Laurens Sluyterman en Ludo van Alst
Advanced Machine Learning @ Radboud University
"""

#%% Import modules & defining constants
from ising_model import IsingModel
from boltzmann import boltzmann_optimiser

N = 10


#%% Main
im = IsingModel(10, True)
boltzmann_optimiser(im)