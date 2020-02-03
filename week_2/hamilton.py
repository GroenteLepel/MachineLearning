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
    :return: gradient of the objective function M(w).
    """
    # compute the outputs from the activations a = x * w
    y = p_star.logistic(data, weights)
    # compute the errors
    errors = labels - y
    # compute the gradient of G(w)
    gradient = np.dot(- data.transpose(), errors)
    return alpha * weights + gradient


def accept(grad):
    if grad < 0:
        return True
    elif np.random.uniform() < np.exp(- grad):
        return True
    else:
        return False


def sample(n_samples: int, data, labels,
           epsilon: float = 0.055, alpha: float = 0.01,
           leap_frog_steps: int = 19):
    """
    Hamilton Monte Carlo method to sample from distribution, finetuned for
    the given p_star_distribution file.
    :param n_samples: amount of steps for sampling.
    :param data: data to use to calculate the energy in p_star_distribution
    to sample the weights to.
    :param labels: labels corresponding to the data.
    :param epsilon: size of the leap frog step. Default is 0.055 (from McKay).
    :param alpha: factor for calculating the energy and gradient. Default is
    0.01 (from McKay).
    :param leap_frog_steps: number of leap frog steps the hamilton sampler
    makes. Default is 19 (from McKay).

    :return: 3 arrays containing the sampled weights, the sampled gradients and
    sampled energies, of shapes (n_steps, 3), (n_steps, 3), (n_steps).
    """
    # initialise arrays for storing
    ws = np.ones(shape=(n_samples, 3))  # weights
    gs = np.zeros(shape=(n_samples, 3))  # gradients
    es = np.zeros(shape=n_samples)  # objective functions

    # initialise the first elements in the saved arrays, w[0] is only ones
    # ws[0] = ws[0] * -3.
    gs[0] = gradient_e(ws[0], data, labels, alpha)
    es[0] = p_star.objective_function(ws[0], data, labels, alpha)

    n_rejected = 0

    print("Starting Longevin sampling method for {} samples.".format(n_samples))
    print("Parameters: epsilon = {}, alpha = {}, leap frog steps = {}"
          .format(epsilon, alpha, leap_frog_steps))
    print("===============")
    print("|", end='')
    for i in range(1, n_samples):
        p = np.random.normal(size=3)  # initial momentum is Normal(0, 1)
        hamiltonian = np.dot(p, p) / 2 + es[i - 1]  # evaluate H(w, p)
        w_new = ws[i - 1]
        g_new = gs[i - 1]

        # make tau 'leapfrog' steps
        for tau in range(leap_frog_steps):
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
            # store new values
            ws[i] = w_new
            gs[i] = g_new
            es[i] = new_energy
        else:
            # use old values
            n_rejected += 1
            ws[i] = ws[i - 1]
            gs[i] = gs[i - 1]
            es[i] = es[i - 1]

        if i / n_samples * 100 % 5 == 0:
            print("█", end='')

    print("|")
    reject_ratio = n_rejected / n_samples
    print("Done. Rejected percentage:", reject_ratio)
    return ws, gs, es, reject_ratio
