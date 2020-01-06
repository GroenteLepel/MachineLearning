#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:30:00 2019

@author: Laurens, Daniël, Ludo
"""


#%% Import packages
import numpy as np 
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


#%% Define functions
def approximate_means(w, theta):
    
    'Calculate the approximate means as a function of w and theta'
    
    def equations(p):
        m1, m2 = p
        return ((m1 - np.tanh(w * (m1 + m2) + theta)), (m2 - np.tanh(w * (m1 + m2) + theta)))
    m1, m2 = fsolve(equations, (0,0))
    return(m1, m2)


def normalization_constant(w, theta):
    
    'Calculate the normalization constant as a function of w and theta'
    
    z = 0
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            z+= np.exp(w * s1 * s2 + theta * (s1 + s2))
    return z



def exact_means(w, theta):
    
    'Calculate the exact means as a function of w and theta'
    
    normalizer = normalization_constant(w, theta)
    exactmean1 = 0
    exactmean2 = 0
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            exactmean1 += s1 * np.exp(w * s1 * s2 + theta * (s1 + s2))
            exactmean2 += s2 * np.exp(w * s1 * s2 + theta * (s1 + s2))
            
    return exactmean1 / normalizer, exactmean2 / normalizer


def exact_chi(w, theta):
    
    'Calculate the exact correlation for a given w and theta' 
    
    m = exact_means(w, theta)
    chi = np.zeros((2,2))
    for i in (0,1):
        for j in (0,1):
            for s in ([1, 1], [1, -1], [-1, 1], [-1, -1]):
                chi[i, j] += s[i] * s[j] * np.exp(w * s[i] * s[j] + theta * \
                    (s[i] + s[i])) / normalization_constant(w, theta)
            chi[i, j] += -m[i] * m[j]
    return chi     


def approximate_chi(w, theta):
    
    'Calculate the approximate linear response for a given w and theta' 
    
    m = exact_means(w, theta)
    chi_inverse = np.zeros((2,2))
    for i in (0,1):
        for j in (0,1):
            if i == j:
                chi_inverse[i, j] = 1 / (1 - m[i] ** 2) - w
            else:
                chi_inverse[i, j] = - w
    return np.linalg.inv(chi_inverse)

    
#%% Make plots for m
thetas = np.linspace(-3, 3, 40)
approximate_firing_rates = np.zeros((40, 2))
exact_firing_rates = np.zeros((40,2))

for i, theta in enumerate(thetas):
    approximate_firing_rates[i] = approximate_means(theta, theta)
    exact_firing_rates[i] = exact_means(theta, theta)

plt.figure(figsize=(15,10))
plt.subplot(1, 2, 1)
plt.plot(thetas, approximate_firing_rates[:,0], label = 'approximation m1')
plt.plot(thetas, exact_firing_rates[:,0], label = 'exact m1')
plt.xlabel('theta')
plt.ylabel('m')
plt.legend(loc = 'lower right')

plt.subplot(1, 2, 2)
plt.plot(thetas, approximate_firing_rates[:,1], label = 'approximation m2')
plt.plot(thetas, exact_firing_rates[:,1], label = 'exact m2')
plt.legend(loc = 'lower right')
plt.xlabel('theta')

plt.show()

#%% Make plots for chi
thetas2 = np.linspace(-0.5, 0.5, 40)
approximate_chis = np.zeros((40, 2, 2))
exact_chis = np.zeros((40, 2, 2))
for i, theta in enumerate(thetas2):
    approximate_chis[i] = approximate_chi(theta, theta)
    exact_chis[i] = exact_chi(theta, theta)

plt.figure(figsize = (12, 12))   
plt.subplot(2, 2, 1)
plt.plot(thetas2, approximate_chis[:,0, 0], label = 'approximation X11')
plt.plot(thetas2, exact_chis[:,0, 0], label = 'exact X11')
plt.legend()
plt.ylabel('Chi')

plt.subplot(2, 2, 2)
plt.plot(thetas2, approximate_chis[:,1, 0], label = 'approximation X21')
plt.plot(thetas2, exact_chis[:,1, 0], label = 'exact X21')
plt.legend()


plt.subplot(2, 2, 3)
plt.plot(thetas2, approximate_chis[:,0, 1], label = 'approximation X12')
plt.plot(thetas2, exact_chis[:,0, 1], label = 'exact X12')
plt.legend()
plt.xlabel('theta')
plt.ylabel('Chi')

plt.subplot(2, 2, 4)
plt.plot(thetas2, approximate_chis[:,1, 1], label = 'approximation X22')
plt.plot(thetas2, exact_chis[:,1, 1], label = 'exact X22')
plt.legend()
plt.xlabel('theta')


plt.show()


