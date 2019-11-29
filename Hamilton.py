#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from p_star_distribution import *
"""
Created on Tue Nov 26 14:45:41 2019

@author: Laurens, Ludo, Daniël
"""

def gradE(weights, data, labels, alpha):
    """
    :return gradient of E
    """
    y = logistic(data, weights)
    sum1=np.array([0,0,0])
    for i in range(0,len(y)):
        for j in range(0,3):
            sum1[j] = sum1[j] + \
            labels[i] * y[i] * data[i,j] * np.exp(-np.dot(weights, data[i])) - \
            (1 - labels[i]) * (1 - y[i]) * data[i, j] * np.exp(np.dot(w, data[i]))
    return sum1 + alpha * weights


def accept(grad):
    if grad < 0:
        return True
    elif np.random.uniform() < np.exp( - grad):
        return True
    else:
        return False    


def objective_function(weights, data, labels, alpha):
    return error_function(weights, data, labels) + alpha * regularizer(weights)


def hamilton_optimizer(nsteps, epsilon, weights, data, labels, alpha):
    ws = np.zeros(shape = (nsteps, 3))
    g = gradE(weights, data, labels, alpha)
    w = weights
    E = objective_function(weights, data, labels, alpha)
    for i in range(nsteps):
        p = np.random.normal(size = 3)
        H = np.dot(p,p) / 2 + E
        wnew = weights
        gnew = gradE(wnew, data, labels, alpha)
        for tau in range(10):
            p = p - epsilon * gnew / 2
            wnew = wnew + epsilon * p
            gnew = gradE(wnew, data, labels, alpha)
            p = p - epsilon * gnew / 2
        Enew = objective_function(wnew, data, labels, alpha)
        Hnew = np.dot(p,p) / 2 + Enew
        dH = Hnew - H
        if accept(dH):
            g = gnew
            w = wnew
            E = Enew
        ws[i] = wnew
    return ws
            

            
            
        