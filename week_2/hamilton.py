#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import week_2.p_star_distribution as p_star
import numpy as np

"""
Created on Tue Nov 26 14:45:41 2019

@author: Laurens, Ludo, Daniël
"""


def gradient_e(weights, data, labels, alpha):
    """
    :return: gradient of E
    """
    y = p_star.logistic(data, weights)
    res = np.array([0, 0, 0])
    for i in range(0, len(y)):
        for j in range(0, 3):
            dot_product = np.dot(weights, data[i])
            res[j] = \
                res[j] + \
                labels[i] * y[i] * data[i, j] * np.exp(-dot_product) - \
                (1 - labels[i]) * (1 - y[i]) * data[i, j] * np.exp(dot_product)
    return res + alpha * weights


def accept(grad):
    if grad < 0:
        return True
    elif np.random.uniform() < np.exp(- grad):
        return True
    else:
        return False


def hamilton_optimizer(n_steps, data, labels,
                       epsilon, alpha):
    """
    Hamilton Monte Carlo method to sample from distribution, finetuned for
    the given p_star_distribution file.
    :param n_steps: amount of steps for sampling.
    :param data: data to use to calculate the energy in p_star_distribution
    to sample the weights to.
    :param labels: labels corresponding to the data.
    :param epsilon: size of the leap frog step.
    :param alpha: factor for calculating the energy and gradient.

    :return: 3 arrays containing the sampled weights, the sampled gradients and
    sampled energies, of shapes (n_steps, 3), (n_steps, 3), (n_steps).
    """
    # initialise arrays for storing
    ws = np.zeros(shape=(n_steps, 3))  # weights
    gs = np.zeros(shape=(n_steps, 3))  # gradients
    es = np.zeros(shape=n_steps)  # objective functions

    gs[0] = gradient_e(ws[0], data, labels, alpha)
    es[0] = p_star.objective_function(ws[0], data, labels, alpha)

    for i in range(1, n_steps):
        p = np.random.normal(size=3)  # initial momentum is Normal(0, 1)
        hamiltonian = np.dot(p, p) / 2 + es[i-1]  # evaluate H(w, p)
        w_new = ws[i-1]
        g_new = gradient_e(w_new, data, labels, alpha)

        # make tau 'leapfrog' steps
        for tau in range(10):
            p = p - epsilon * g_new / 2  # make half-step in p
            w_new = w_new + epsilon * p  # make step in w
            g_new = gradient_e(w_new, data, labels, alpha)  # find new gradient
            p = p - epsilon * g_new / 2  # make half-step in p

        # find new value of hamiltonian
        new_energy = p_star.objective_function(w_new, data, labels, alpha)
        new_hamiltonian = np.dot(p, p) / 2 + new_energy

        # decide whether to accept
        grad_hamiltonian = new_hamiltonian - hamiltonian
        if accept(grad_hamiltonian):
            energy = new_energy

        # store value
        ws[i] = w_new
        gs[i] = g_new
        es[i] = new_energy

    return ws, gs, es
