#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:34:25 2019

@author: Laurens, Ludo, Daniël

"""
import numpy as np

p = np.zeros((3,3,3))

p[0,:,:] =[[0.0057,    0.0103,    0.0561],
            [0.0027,    0.0102,    0.0026],
            [0.0713,    0.0154,    0.0456]]


p[1,:,:] = [[0.0398,    0.0012,    0.0318],
            [0.0649,    0.0711,    0.0528],
            [0.0138,    0.0489,    0.0547]]


p[2,:,:] = [[0.0296,    0.0276,    0.0379],
            [0.0617,    0.0453,    0.0646],
            [0.0432,    0.0878,    0.0034]]

q_xy = np.ones((3,3)) / 9
q_z =  np.ones(3) / 3

def KL(q_xy, q_z, p):
    
    'Calculating the KL divergence between our approximate q and p' 
    
    KL = 0
    for i in range(0,2):
        for j in range(0,2):
            for k in range(0,2):
                KL += q_xy[i, j] * q_z[k] * np.log(q_xy[i, j] * q_z[k] / p[i, j, k])
    return KL


Delta_KL = 5
KLs = []
while Delta_KL > 0.0000001:
    KLold = KL(q_xy, q_z, p)
    KLs.append(KLold)
    for k in (0,2):
        q_z[k] = np.exp(np.sum(((np.log(p[:,:,k]) * q_xy * q_z[k]))))
        q_z = q_z / np.sum(q_z)
        KLs.append(KL(q_xy, q_z, p))
    for i in (0, 2):
        for j in (0, 2):
            q_xy[i, j] = np.exp(np.sum((np.log(p[i, j, :]) * q_z * q_xy[i, j])))
            q_xy = q_xy / np.sum(q_xy)
        KLs.append(KL(q_xy, q_z, p))
    KLnew = KL(q_xy, q_z, p)
    KLs.append(KLnew)
    Delta_KL = KLold - KLnew
    
KL(q_xy, q_z, p)
