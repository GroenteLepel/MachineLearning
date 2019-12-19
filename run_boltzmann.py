#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:15:36 2019

@author: Daniel Kok, Laurens Sluyterman en Ludo van Alst
Advanced Machine Learning @ Radboud University
"""

#%% Import modules & defining constants
from ising_model import IsingModel
import boltzmann

N = 10


#%% Main
im = IsingModel(N, frustrated = True, threshold = True)
states = boltzmann.generate_state_set(im,5)
print(boltzmann.LLH(im,states))