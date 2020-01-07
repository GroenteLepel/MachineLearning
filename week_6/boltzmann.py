#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:09:58 2019

@author: Daniel Kok, Laurens Sluyterman en Ludo van Alst
Advanced Machine Learning @ Radboud University
Assignment: Boltzmann
"""

# %% Importing modules
import itertools

import numpy as np
from week_3.ising_model import IsingModel
from week_6.ising_ensemble import IsingEnsemble
import copy


# %% Defining functions
def boltzmann_optimiser(ie: IsingEnsemble, eta=0.5):
    """
        Optimise the Boltzmann machine. Works as follows:
            - calculate the gradient of LLH w.r.t. thresholds and weights
            - update thresholds and weights: += eta * grad(LLH)
            - TODO: in which order should we update, all at the same time or
                    one by one (I think one by one because in the former case
                    it could not converge)

        :return:
    """
    # Initialise criterion value
    dw_abs, dtheta = 1, 1
    likelihood = np.zeros(int(1e5))
    cnt = 0

    while not (dw_abs == 0 and dtheta == 0):
        print(cnt)
        print(dw_abs, dtheta)
        old_w = copy.copy(ie.coupling_matrix)
        old_t = copy.copy(ie.threshold_vector)

        for i in range(ie.n_spins):
            for j in range(ie.n_spins):
                llh_grad_w = \
                    ie.expectation_matrix_c[i][j] - \
                    expectation(ie.coupling_matrix, ie.threshold_vector, i, j)

                ie.coupling_matrix[i][j] += eta * llh_grad_w

            llh_grad_t = \
                    ie.expectation_vector_c[i] - \
                    expectation(ie.coupling_matrix, ie.threshold_vector, i)

            ie.threshold_vector[i] += eta * llh_grad_t

        dw_abs = np.abs(ie.coupling_matrix - old_w).sum()
        dtheta = np.abs(ie.threshold_vector - old_t).sum()

        likelihood[cnt] = ie.LLH()
        print(likelihood[cnt])
        cnt += 1

    return likelihood[likelihood < 0]


def gen_all_possible_states(n_spins):
    lst = list(map(list, itertools.product([-1, 1], repeat=n_spins)))
    return np.array(lst)


def expectation(coupling_matrix, threshold_vector, i: int, j: int = -1):
    states = gen_all_possible_states(len(threshold_vector))

    dummy_state = IsingModel(len(threshold_vector), True, True)
    dummy_state.coupling_matrix = coupling_matrix
    dummy_state.threshold_vector = threshold_vector

    if j == -1:
        sum_val = 0
        for state in states:
            dummy_state.state = state
            sum_val += state[i] * dummy_state.p()
    else:
        sum_val = 0
        for state in states:
            dummy_state.state = state
            sum_val += state[i] * state[j] * dummy_state.p()

    return sum_val
