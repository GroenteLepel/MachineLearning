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
import week_6.all_states as all_states
from week_6.ising_ensemble import IsingEnsemble
import copy


# %% Defining functions
def boltzmann_optimiser(ie: IsingEnsemble, eta=0.4, output: bool=False):
    """
        Optimise the Boltzmann machine. Works as follows:
            - calculate the gradient of LLH w.r.t. thresholds and weights
            - update thresholds and weights: += eta * grad(LLH)

        :return:
    """
    # Initialise criterion value
    dw_abs, dtheta = 1, 1
    likelihood = np.zeros(int(1e5))
    cnt = 0
    diff_llh = 123456
    likelihood[-1] = -1e5
    while diff_llh > 1e-4:
        if output:
            print("---")
            print("Iteration", cnt)
            print("[c", end='')
        old_w = copy.copy(ie.coupling_matrix)
        old_t = copy.copy(ie.threshold_vector)

        for i in range(ie.n_spins):
            if output:
                print("==", end='')
            for j in range(i + 1, ie.n_spins):
                llh_grad_w = \
                    ie.expectation_matrix_c[i][j] - \
                    expectation(ie.coupling_matrix, ie.threshold_vector,
                                ie.normalisation_constant, i, j)

                ie.coupling_matrix[i][j] += eta * llh_grad_w
                ie.coupling_matrix[j][i] += eta * llh_grad_w
                ie.update_normalisation_constants()

            llh_grad_t = \
                    ie.expectation_vector_c[i] - \
                    expectation(ie.coupling_matrix, ie.threshold_vector,
                                ie.normalisation_constant, i)

            ie.threshold_vector[i] += eta * llh_grad_t
            ie.update_normalisation_constants()


        dw_abs = np.abs(ie.coupling_matrix - old_w).sum()
        dtheta = np.abs(ie.threshold_vector - old_t).sum()

        likelihood[cnt] = ie.LLH()
        if output:
            print("3]")
            print("dw_abs:", dw_abs, "dtheta:", dtheta)
            print("Log-likelihood:", likelihood[cnt])
            print("---\n")
        diff_llh = likelihood[cnt] - likelihood[cnt - 1]
        cnt += 1

    return likelihood[:cnt]


def expectation(coupling_matrix, threshold_vector, normalisation_constant,
                i: int, j: int = -1):
    states = all_states.gen_all_possible_states(len(threshold_vector))

    dummy_state = IsingModel(len(threshold_vector), True, True)
    dummy_state.coupling_matrix = coupling_matrix
    dummy_state.threshold_vector = threshold_vector
    dummy_state.normalisation_constant = normalisation_constant

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
