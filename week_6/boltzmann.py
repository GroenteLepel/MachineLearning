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
import copy
import sys

sys.path.append("../week_3")
from ising_model import IsingModel
sys.path.append("../week_6")
import all_states as all_states
from ising_ensemble import IsingEnsemble


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


def constraint(method: str, diff_llh = 1e10, dw_avg = 1e10, dtheta_avg = 1e10, iterations = -1, iterations_bound: int = 100):
    if method == 'exact':
        return diff_llh > 1e-3
    else:
        return iterations < iterations_bound
        #return (not (dw_avg < 1e-1 and dtheta_avg < 1e-2)) and iterations < iterations_bound


def optimise(ie: IsingEnsemble, eta: float,
             method: str, output: bool, n_MC_samples: int = 500, iterations_bound: int = 100):
    # Initialise criterion value
    likelihood = np.zeros(int(1e5))
    cnt = 0
    diff_llh = 123456
    likelihood[-1] = -1e5
    last_sample = []
    dw_avg = 1e10
    dtheta_avg = 1e10

    while constraint(method, diff_llh, dw_avg = dw_avg, dtheta_avg = dtheta_avg, iterations = cnt, iterations_bound = iterations_bound):
        if output:
            print("---")
            print("Iteration", cnt+1)
            print("[c", end='')
        old_w = copy.copy(ie.coupling_matrix)
        old_t = copy.copy(ie.threshold_vector)

        for i in range(ie.n_spins):
            if output:
                print("==", end='')
            for j in range(i + 1, ie.n_spins):
                double_expec, last_sample = give_expectation(method, ie, i, j, n_MC_samples = n_MC_samples, last_sample = last_sample)

                llh_grad_w = ie.expectation_matrix_c[i][j] - double_expec

                ie.coupling_matrix[i][j] += eta * llh_grad_w
                ie.coupling_matrix[j][i] += eta * llh_grad_w
                if ie.n_spins <= 10:
                    ie.update_normalisation_constants()

            single_expec, last_sample = give_expectation(method, ie, i, n_MC_samples = n_MC_samples, last_sample = last_sample)
            
            llh_grad_t = ie.expectation_vector_c[i] - single_expec

            ie.threshold_vector[i] += eta * llh_grad_t
            if ie.n_spins <= 10:
                ie.update_normalisation_constants()

        dw_avg = np.abs(ie.coupling_matrix - old_w).sum()/len(ie.threshold_vector)**2
        dtheta_avg = np.abs(ie.threshold_vector - old_t).sum()/len(ie.threshold_vector)

        if ie.n_spins <= 10:
            likelihood[cnt] = ie.LLH()

        if output:
            print("3]")
            if ie.n_spins <= 10:
                print("Log-likelihood:", likelihood[cnt])
            if method == 'mc':
                print("dw_avg:", dw_avg, "dtheta_avg:", dtheta_avg)
                eta *= 0.98
                print('eta', eta)
            print("---\n")
        
        if method == 'exact':
            diff_llh = np.abs(likelihood[cnt] - likelihood[cnt - 1])
            
        cnt += 1
    
    if ie.n_spins <= 10:
        return ie.coupling_matrix, ie.threshold_vector, likelihood[:cnt], cnt
    else:
        return ie.coupling_matrix, ie.threshold_vector, cnt


def give_expectation(method: str, ie: IsingEnsemble, i: int, j: int = -1, n_MC_samples: int = 500, last_sample = []):
    if method == '' or method == 'exact':
        expec = exact_expectation(ie.coupling_matrix,
                                  ie.threshold_vector,
                                  ie.normalisation_constant, i, j)
    elif method == 'mc':
        expec, last_sample = get_that_motherfucking_mc_exp_value(ie, i, j, n_MC_samples, last_sample = last_sample)
    else:
        raise ValueError(
            "No proper method given. Choose 'exact' or 'mc'.")

    return expec, last_sample


#def monte_carlo_optimise(ie: IsingEnsemble, eta: float, output: bool):
#    likelihood = np.zeros(int(1e5))
#
#    expectation_value = get_that_motherfucking_mc_exp_value(ie)


def get_that_motherfucking_mc_exp_value(ie: IsingEnsemble,
                                        i_index: int, j_index: int = -1,
                                        n_MC_samples: int = 500, last_sample = []):
    if last_sample == []:
        if np.average(ie.expectation_vector_c) < -0.5:
            last_sample = np.array([-1] * ie.n_spins)
        else:
            last_sample = np.random.choice([-1,1], size=(ie.n_spins,))
            
    expectation_value = 0
    for i in range(1, n_MC_samples + 1):
        new_sample = gen_new_state(ie, last_sample)
        if j_index >= 0:
            expectation_value += new_sample[i_index] * new_sample[j_index]
        else:
            expectation_value += new_sample[i_index]

        last_sample = copy.deepcopy(new_sample)

    return expectation_value / n_MC_samples, last_sample


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
                  
                  
def read_salamander_expectations(filepath: str, n_spins: int):
    expectation_vector_c = np.zeros(n_spins)
    expectation_matrix_c = np.zeros((n_spins,n_spins))
    
    with open(filepath, 'r') as f:
        firstline = f.readline()
        for index, elt in enumerate(firstline.split()):
            expectation_vector_c[index] = float(elt)
        f.readline()
        for index1, line in enumerate(f):
            for index2, elt in enumerate(line.split()):
                expectation_matrix_c[index1][index2] = float(elt)
                
    return expectation_matrix_c, expectation_vector_c
                

def mean_field_estimate(expectation_matrix_c, expectation_vector_c):
    n_spins = len(expectation_vector_c)
    m = expectation_vector_c
    C = expectation_matrix_c - np.outer(m, m)
    C_inv = np.linalg.inv(C)
    coupling_matrix = 1/(1-np.tile(m,(n_spins,1))**2) * np.identity(n_spins) - C_inv
    threshold_vector = np.arctanh(m) - np.inner(coupling_matrix, m)
    
    return coupling_matrix, threshold_vector