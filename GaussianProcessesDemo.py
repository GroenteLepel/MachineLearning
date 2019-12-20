#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:37:29 2019

@author: laurens
"""

import numpy as np
import matplotlib.pyplot as plt

def mean_function(x):
   # return x**2
   return 0


def covariance_function(x, y):
  # return x*y
    return 0.5 * np.exp(-(x - y) ** 2)


N = 50
mean = np.zeros(N)
K = np.zeros((N,N))
x = np.linspace(0, 10, N)

for i in range(0,N):   
    mean[i] = mean_function(x[i])
    for j in range(0, N):
        K[i, j] = covariance_function(x[i], x[j])


y1 = np.random.multivariate_normal(mean, K)
y2 = np.random.multivariate_normal(mean, K)
y3 = np.random.multivariate_normal(mean, K)

plt.plot(x, y1, 'o')
plt.plot(x, y2, 'o')
plt.plot(x, y3, 'o')
plt.show()