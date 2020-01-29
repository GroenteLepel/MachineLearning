#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from week_2.p_star_distribution import *

"""
Created on Tue Nov 26 14:45:41 2019

@author: Laurens, Ludo, Daniël
"""


def gradient_e(weights, data, labels, alpha):
    """
    :return gradient of E
    """
    y = logistic(data, weights)
    sum1 = np.array([0, 0, 0])
    for i in range(0, len(y)):
        for j in range(0, 3):
            sum1[j] = sum1[j] + \
                      labels[i] * y[i] * data[i, j] * np.exp(
                -np.dot(weights, data[i])) - \
                      (1 - labels[i]) * (1 - y[i]) * data[i, j] * np.exp(
                np.dot(w, data[i]))
    return sum1 + alpha * weights


def accept(grad):
    if grad < 0:
        return True
    elif np.random.uniform() < np.exp(- grad):
        return True
    else:
        return False


def objective_function(weights, data, labels, alpha):
    return error_function(weights, data, labels) + alpha * regularizer(weights)


def hamilton_optimizer(nsteps, epsilon, weights, data, labels, alpha):
    ws = np.zeros(shape=(nsteps, 3))
    g = gradient_e(weights, data, labels, alpha)
    w = weights
    energy = objective_function(weights, data, labels, alpha)
    for i in range(nsteps):
        p = np.random.normal(size=3)
        hamiltonian = np.dot(p, p) / 2 + energy
        w_new = weights
        g_new = gradient_e(w_new, data, labels, alpha)
        for tau in range(10):
            p = p - epsilon * g_new / 2
            w_new = w_new + epsilon * p
            g_new = gradient_e(w_new, data, labels, alpha)
            p = p - epsilon * g_new / 2
        new_energy = objective_function(w_new, data, labels, alpha)
        new_hamiltonian = np.dot(p, p) / 2 + new_energy
        grad_hamiltonian = new_hamiltonian - hamiltonian
        if accept(grad_hamiltonian):
            g = g_new
            w = w_new
            energy = new_energy
        ws[i] = w_new
    return ws
