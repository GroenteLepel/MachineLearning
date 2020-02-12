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
import os

sys.path.append("../week_3")
from ising_model import IsingModel
sys.path.append("../week_6")
import all_states as all_states
from ising_ensemble import IsingEnsemble


# %% Defining functions
def n_salamander_states_to_ie(filepath: str, n_salamander_states: int):
    state_set = np.zeros(shape=(n_salamander_states, 160))
    ie = IsingEnsemble(n_salamander_states, 160, frustrated = True)
    
    with open(filepath, 'r') as f:
        for neuronindex, line in enumerate(f):
            linesplit = line.split()
            for stateindex, elt in enumerate(linesplit[:n_salamander_states]):
                eltint = int(elt)
                if eltint == 0:
                    eltint = -1
                state_set[stateindex][neuronindex] = eltint
                
    for stateindex, state in enumerate(state_set):
        ie.state_set[stateindex].state = state

    return ie


def constraint(method: str, diff_llh = 1e10, dw_avg = 1e10, dtheta_avg = 1e10, iterations = -1, iterations_bound: int = 100):
    if method == 'exact':
        return diff_llh > 1e-2
    else:
        return iterations < iterations_bound
        return (not (dw_avg < 1e-1 and dtheta_avg < 1e-2)) and iterations < iterations_bound


def print_output(method: str, iteration: int, current_llh: float, total_spins: int, i: int, j: int = -1):
    os.system('clear')
    if iteration == 0:
        current_llh = 'not yet calculated.'
    print("---")
    print("Method:\t\t", method)
    print("Iteration:\t", iteration + 1)
    print("Current log-likelihood:\t", current_llh)
    print("Updating spin {0:d}/{1:d}".format(i,total_spins))
    print("w[{0:d}][_] \t theta[{1:d}]".format(i,i))
        #print("Updating coupling matrix: w[{0:d}][{1:d}]".format(i,j))
        #print("Updating threshold vector: theta[{0:d}]".format(i))
    print("---")


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


def optimise(ie: IsingEnsemble, eta: float,
             method: str, output: bool, burn_in_iterations: int = -1, n_MC_state_samples: int = 500, iterations_bound: int = 100):
    # Initialise criterion value
    likelihood = np.zeros(int(1e5))
    cnt = 0
    diff_llh = 123456
    likelihood[-1] = -1e5
    dw_avg = 1e10
    dtheta_avg = 1e10
    last_sample = []

    while constraint(method, diff_llh, dw_avg = dw_avg, dtheta_avg = dtheta_avg, iterations = cnt, iterations_bound = iterations_bound):
        old_w = copy.copy(ie.coupling_matrix)
        old_t = copy.copy(ie.threshold_vector)

        for i in range(ie.n_spins):
            if output:
                print_output(method, cnt, likelihood[cnt-1], ie.n_spins, i)
            for j in range(i + 1, ie.n_spins):
                if method == 'exact':
                    ie.update_normalisation_constants()
                    
                double_expec, last_sample = give_expectation(method, ie, i, j, burn_in_iterations = burn_in_iterations, n_MC_state_samples = n_MC_state_samples, last_sample = last_sample)

                llh_grad_w = ie.expectation_matrix_c[i][j] - double_expec

                ie.coupling_matrix[i][j] += eta * llh_grad_w
                ie.coupling_matrix[j][i] += eta * llh_grad_w
                
            if method == 'exact':
                ie.update_normalisation_constants()
                
            single_expec, last_sample = give_expectation(method, ie, i, burn_in_iterations = burn_in_iterations, n_MC_state_samples = n_MC_state_samples, last_sample = last_sample)
            
            llh_grad_t = ie.expectation_vector_c[i] - single_expec

            ie.threshold_vector[i] += eta * llh_grad_t

        dw_avg = np.abs(ie.coupling_matrix - old_w).sum()/len(ie.threshold_vector)**2
        dtheta_avg = np.abs(ie.threshold_vector - old_t).sum()/len(ie.threshold_vector)

        if method == 'exact' or ie.n_spins <= 10:
            ie.update_normalisation_constants()
        else:
            ie.normalisation_constant, last_sample = estimate_normalisation_constant(ie, burn_in_iterations = burn_in_iterations, last_sample = last_sample)
            ie.update_normalisation_constants(normalisation_constant = ie.normalisation_constant)
        likelihood[cnt] = ie.LLH()

#        if output:
#            print("3]")
##            if ie.n_spins <= 10:
#            print("Log-likelihood:", likelihood[cnt])
##            if method == 'mc':
##                print("dw_avg:", dw_avg, "dtheta_avg:", dtheta_avg)
##                eta *= 0.98
##                print('eta', eta)
#            print("---\n")
        
#        if method == 'exact':
        diff_llh = np.abs(likelihood[cnt] - likelihood[cnt - 1])
            
        cnt += 1
    
#    if ie.n_spins <= 10:
    return ie.coupling_matrix, ie.threshold_vector, likelihood[:cnt], cnt
#    else:
#        return ie.coupling_matrix, ie.threshold_vector, cnt
    

#def LLH_estimate(normalisation_constant, state_set, coupling_matrix, threshold_vector):
#    loglikelihood = 0
#    for state in state_set:
#        loglikelihood += np.log(1./ normalisation_constant * pstar_BG(state, coupling_matrix, threshold_vector))
#    return 1. / len(state_set) * loglikelihood


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
    
    
def estimate_normalisation_constant(ie: IsingEnsemble, burn_in_iterations: int = -1, last_sample = []):
    if 2**ie.n_spins < 1e4:
        n_MC_NC_samples = int(0.5 * 2**ie.n_spins)
    else:
        n_MC_NC_samples = int(ie.n_spins**2)
    
    if last_sample == []:
        if np.average(ie.expectation_vector_c) < -0.5:
            last_sample = np.array([-1] * ie.n_spins)
        else:
            last_sample = np.random.choice([-1,1], size=(ie.n_spins,))
    if burn_in_iterations == -1:
        burn_in_iterations = ie.n_spins
            
    normalisation_constant_estimate = 0
    for iteration in range(burn_in_iterations + n_MC_NC_samples):
        new_sample = gen_new_state(ie, last_sample)
        if iteration >= burn_in_iterations:
            normalisation_constant_estimate += pstar_BG(new_sample, ie.coupling_matrix, ie.threshold_vector)

        last_sample = copy.deepcopy(new_sample)

    return normalisation_constant_estimate, last_sample
#    n_spins = len(ie.threshold_vector)
#    if 2**n_spins < 1e4:
#        n_MC_NC_samples = int(0.5 * 2**n_spins)
#    else:
#        n_MC_NC_samples = int(n_spins**2)
#    scaling_factor = 2**n_spins / n_MC_NC_samples
#    
#    normalisation_constant_estimate = 0
#    for _ in range(n_MC_NC_samples):
#        sample_state = np.random.choice([-1, 1], size=n_spins)
#        normalisation_constant_estimate += pstar_BG(sample_state, ie.coupling_matrix, ie.threshold_vector)
#    
#    return normalisation_constant_estimate * scaling_factor


def pstar_BG(state, coupling_matrix, threshold_vector):
    """
    Computes the Boltzmann-Gibbs probability without dividing by the
    normalisation constant, called pstar. This is needed to estimate the
    normalisation constant.
    """
    ising_energy = -0.5 * np.dot(state, np.dot(coupling_matrix, state)) \
           - np.dot(threshold_vector, state)

    return np.exp(-ising_energy)


def give_expectation(method: str, ie: IsingEnsemble, i: int, j: int = -1, burn_in_iterations: int = 200, n_MC_state_samples: int = 500, last_sample = []):
    if method == '' or method == 'exact':
        expec = exact_expectation(ie.coupling_matrix,
                                  ie.threshold_vector,
                                  ie.normalisation_constant, i, j)
    elif method == 'mc':
        expec, last_sample = get_that_motherfucking_mc_exp_value(ie, i, j, burn_in_iterations, n_MC_state_samples)
    else:
        raise ValueError(
            "No proper method given. Choose 'exact' or 'mc'.")

    return expec, last_sample


#def monte_carlo_optimise(ie: IsingEnsemble, eta: float, output: bool):
#    likelihood = np.zeros(int(1e5))
#
#    expectation_value = get_that_motherfucking_mc_exp_value(ie)
    

def get_that_motherfucking_mc_exp_value_v2(ie: IsingEnsemble,
                                        i_index: int, j_index: int = -1,
                                        n_MC_state_samples: int = 500):
    '''
    This version of the function randomly (uniform) takes some random 
    samples and averages them to find the expectation value.
    '''
    expectation_value = 0
    scaling_factor = 2**ie.n_spins / n_MC_state_samples
    coupling_matrix = ie.coupling_matrix
    threshold_vector = ie.threshold_vector
    
    for _ in range(n_MC_state_samples):
        sample = np.random.choice([-1,1], size=(ie.n_spins,))
        if j_index >= 0:
            expectation_value += sample[i_index] * sample[j_index] * pstar_BG(sample, coupling_matrix, threshold_vector)
        else:
            expectation_value += sample[i_index] * pstar_BG(sample, coupling_matrix, threshold_vector)

    return 1./ ie.normalisation_constant * expectation_value * scaling_factor


def get_that_motherfucking_mc_exp_value(ie: IsingEnsemble,
                                        i_index: int, j_index: int = -1, burn_in_iterations: int = 200,
                                        n_MC_state_samples: int = 500, last_sample = []):
    '''
    This version of the function samples in a neighbourhood of the previous
    sample after a burn-in phase.
    '''
    if last_sample == []:
        if np.average(ie.expectation_vector_c) < -0.5:
            last_sample = np.array([-1] * ie.n_spins)
        else:
            last_sample = np.random.choice([-1,1], size=(ie.n_spins,))
    if burn_in_iterations == -1:
        burn_in_iterations = ie.n_spins
            
    expectation_value = 0
    for iteration in range(burn_in_iterations + n_MC_state_samples):
        new_sample = gen_new_state(ie, last_sample)
        if iteration >= burn_in_iterations:
            if j_index >= 0:
                expectation_value += new_sample[i_index] * new_sample[j_index]
            else:
                expectation_value += new_sample[i_index]

        last_sample = copy.deepcopy(new_sample)

    return expectation_value / n_MC_state_samples, last_sample


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
    coupling_matrix = np.multiply(1/(1-np.tile(m,(n_spins,1))**2), np.identity(n_spins)) - C_inv
    threshold_vector = np.arctanh(m) - np.inner(coupling_matrix, m)
    
    return coupling_matrix, threshold_vector