#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:09:58 2019

@author: Daniel Kok, Laurens Sluyterman en Ludo van Alst
Advanced Machine Learning @ Radboud University
Assignment: Boltzmann
"""


#%% Importing modules
import numpy as np
from week_3.ising_model import IsingModel
import copy


#%% Defining functions
def generate_state_set(isingmodel: IsingModel, size: int):
    # Generate a set of states
    state_set = np.array([isingmodel._generate_spin_state() for _ in range(size)])
    return state_set

def LLH(isingmodel: IsingModel, states):
    # Calculate the log-likelihood for the set of states (par 2.5 of the reader)
    P = len(states)
    loglikelihood = 0
    for state in states:
        isingmodel_dummy = copy.deepcopy(isingmodel)
        isingmodel_dummy.state = state
        loglikelihood += np.log(isingmodel_dummy.p())
    return 1./P * loglikelihood

def expectations(isingmodel: IsingModel, states):
    # Calculate the fixed expectation values <s_i> and <s_i s_j>
    # Returns a vector and a matrix respectively
    dummy = copy.deepcopy(isingmodel)
    
    expectation_s = np.zeros(isingmodel.n)
    expectation_ss = np.zeros((isingmodel.n, isingmodel.n))
    
    for k in range(isingmodel.n + 1):
        # We want to loop through all possible spin configurations of length n
        # To do this I use all flip combinations possible of length 1-n
        flip_combinations = isingmodel.generate_flip_combinations(k)
        for flip_combination in flip_combinations:
            dummy.state = np.ones(isingmodel.n)
            if k != 0:
                # for k==0, flip_combination is an empty list and this gives
                # problems in this step. we do not have to flip in this case.
                dummy.state[flip_combination] *= -1
                
            for i in range(isingmodel.n):
                expectation_s[i] += dummy.state[i] * dummy.p()
                for j in range(isingmodel.n):
                    expectation_ss[i][j] += dummy.state[i] * dummy.state[j] * dummy.p()
            
    return [expectation_s, expectation_ss]

def expectation_si_clamped(states, index):
    P = len(states)
    return 1./P * np.sum(states[:, index])

def expectation_sisj_clamped(states, index1, index2):
    P = len(states)
    return 1./P * np.sum(states[:, index1] * states[:, index2])

def boltzmann_optimiser(isingmodel: IsingModel, states, eta):
    """
        Optimise the Boltzmann machine. Works as follows:
            - calculate the gradient of LLH w.r.t. thresholds and weights
            - update thresholds and weights: += eta * grad(LLH)
            - TODO: in which order should we update, all at the same time or
                    one by one (I think one by one because in the former case
                    it could not converge)

        :return: weights, thresholds
    """
    [expectation_s, expectation_ss] = expectations(isingmodel, states)
    
    return weights, thresholds