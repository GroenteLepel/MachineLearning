#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:09:58 2019

@author: Daniel Kok, Laurens Sluyterman en Ludo van Alst
Advanced Machine Learning @ Radboud University
Assignment: Boltzmann
"""

# %% Importing modules
import numpy as np
from week_3.ising_model import IsingModel
import week_6.all_states as all_states
from week_6.ising_ensemble import IsingEnsemble
import copy


# %% Defining functions
def exact_expectation(coupling_matrix, threshold_vector, normalisation_constant,
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


def boltzmann_optimiser(ie: IsingEnsemble,
                        eta=0.4, method: str = '',
                        output: bool = False):
    """
        Optimise the Boltzmann machine. Works as follows:
            - calculate the gradient of LLH w.r.t. thresholds and weights
            - update thresholds and weights: += eta * grad(LLH)

        :return:
    """

    if method == '':
        if ie.n_spins > 10:
            likelihood = optimise(ie, eta, method='mc', output=output)
        else:
            likelihood = optimise(ie, eta, method='exact', output=output)
    else:
        likelihood = optimise(ie, eta, method=method, output=output)

    return likelihood


def constraint(method: str, diff_llh):
    if method == 'exact':
        return diff_llh > 1e-4
    else:
        return diff_llh > 1e-4


def optimise(ie: IsingEnsemble, eta: float,
             method: str, output: bool):
    # Initialise criterion value
    likelihood = np.zeros(int(1e5))
    cnt = 0
    diff_llh = 123456
    dw_abs, dtheta = 1, 1
    likelihood[-1] = -1e5

    while constraint(method, diff_llh):
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
                double_expec = give_expectation(method, ie, i, j)

                llh_grad_w = ie.expectation_matrix_c[i][j] - double_expec

                ie.coupling_matrix[i][j] += eta * llh_grad_w
                ie.coupling_matrix[j][i] += eta * llh_grad_w
                # if method == 'exact':
                ie.update_normalisation_constants()

            single_expec = give_expectation(method, ie, i)

            llh_grad_t = ie.expectation_vector_c[i] - single_expec

            ie.threshold_vector[i] += eta * llh_grad_t
            # if method == 'exact':
            ie.update_normalisation_constants()

        dw_abs = np.abs(ie.coupling_matrix - old_w).sum()
        dtheta = np.abs(ie.threshold_vector - old_t).sum()

        if ie.n_spins <= 10:
            likelihood[cnt] = ie.LLH()

        if output:
            print("3]")
            print("dw_abs:", dw_abs, "dtheta:", dtheta)
            print("Log-likelihood:", likelihood[cnt])
            print("---\n")
        diff_llh = np.abs(likelihood[cnt] - likelihood[cnt - 1])
        cnt += 1
    return likelihood[:cnt]


def give_expectation(method: str, ie: IsingEnsemble, i: int, j: int = -1):
    if method == '' or method == 'exact':
        expec = exact_expectation(ie.coupling_matrix,
                                  ie.threshold_vector,
                                  ie.normalisation_constant, i, j)
    elif method == 'mc':
        expec = get_that_motherfucking_mc_exp_value(ie, i, j)
    else:
        raise ValueError(
            "No proper method given. Choose 'exact' or 'mc'.")

    return expec


def monte_carlo_optimise(ie: IsingEnsemble, eta: float, output: bool):
    likelihood = np.zeros(int(1e5))

    expectation_value = get_that_motherfucking_mc_exp_value(ie)


def get_that_motherfucking_mc_exp_value(ie: IsingEnsemble,
                                        i_index: int, j_index: int = -1,
                                        n_samples: int = 500):
    old_sample = np.random.choice([-1, 1], size=(ie.n_spins,))
    expectation_value = 0
    for i in range(1, n_samples + 1):
        new_sample = gen_new_state(ie, old_sample)
        if j_index >= 0:
            expectation_value += new_sample[i_index] * new_sample[j_index]
        else:
            expectation_value += new_sample[i_index]

        old_sample = copy.deepcopy(new_sample)

    return expectation_value / n_samples


def gen_new_state(ie: IsingEnsemble, old_state):
    rand_element = np.random.randint(ie.n_spins)
    accept_ratio = state_probability_fraction(ie, old_state,
                                              rand_element)
    new_state = copy.deepcopy(old_state)
    if accept_ratio >= 1 or np.random.uniform() <= accept_ratio:
        new_state[rand_element] *= -1

    return new_state


def state_probability_fraction(ie: IsingEnsemble, old_state, index: int):
    return np.exp(-2 * old_state[index] *
                  (np.dot(ie.coupling_matrix[index, :], old_state) +
                   ie.threshold_vector[index])
                  )
