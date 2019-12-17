#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:09:58 2019

@author: Daniel Kok, Laurens Sluyterman en Ludo van Alst
Advanced Machine Learning @ Radboud University
"""

#%% Importing modules
import numpy as np
from ising_model import IsingModel


#%% Defining functions
def boltzmann_optimiser(isingmodel: IsingModel):
    # No mean field for the first bullet of the exercise!
    # We need to do it exactly