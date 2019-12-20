#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:34:25 2019

@author: Laurens, Ludo, Daniël

"""

#%% Importing modules
import numpy as np
import matplotlib.pyplot as plt


#%% Initializing distributions
p = np.zeros((3,3,3))

p[:,:,0] =[[0.0057,    0.0103,    0.0561],
            [0.0027,    0.0102,    0.0026],
            [0.0713,    0.0154,    0.0456]]


p[:,:,1] = [[0.0398,    0.0012,    0.0318],
            [0.0649,    0.0711,    0.0528],
            [0.0138,    0.0489,    0.0547]]


p[:,:,2] = [[0.0296,    0.0276,    0.0379],
            [0.0617,    0.0453,    0.0646],
            [0.0432,    0.0878,    0.0034]]

q_xy = np.ones((3,3)) / 9
q_z =  np.ones(3) / 3

#%% Defining functions
def KL(Q_xy, Q_z, P):
    
    'Calculating the KL divergence between our approximate q and p' 
    
    KL = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                KL += Q_xy[i, j] * Q_z[k] * np.log(Q_xy[i, j] * Q_z[k] / P[i, j, k])
    return KL


#%% Main algo
Delta_KL = 5
KLs = [KL(q_xy, q_z, p)]

while Delta_KL > 0.00000001:
    KLold = KLs[-1]
    
    for k in range(3):
        q_z[k] = np.exp(np.sum(((np.log(p[:,:,k]) * q_xy))))
    q_z /= np.sum(q_z)

    for i in range(3):
        for j in range(3):
            q_xy[i, j] = np.exp(np.sum((np.log(p[i, j, :]) * q_z)))
    q_xy /= np.sum(q_xy)
    
    KLnew = KL(q_xy, q_z, p)
    KLs.append(KLnew)
    Delta_KL = KLold - KLnew


plt.plot(KLs)
plt.xlabel('iteration')
plt.ylabel('KL(q|p)')
plt.show()
   
print("final KL = ", KL(q_xy, q_z, p))
