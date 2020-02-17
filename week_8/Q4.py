#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:50:31 2020

@author: laurens
"""

import numpy as np
import matplotlib.pyplot as plt
def u(t, x, T , nu):
    return (np.tanh(x / (nu * (T - t))) - x ) / (T - t)


T = 100
nu = 1
n = 500

for j in range(0,10):
    times =  np.arange(T, step = T/n)
    x = [0]
    time = [0]
    control = np.zeros(n)
    for  i in range(0,n-1):
        dt = T / n
        control[i] = u(time[i], x[i], T, nu)
        dx = control[i] * dt + np.random.normal(0, np.sqrt(nu * dt))
        x.append(x[i] + dx)
        time.append(i * dt)
    plt.title('T = '+str(T)+', nu = '+str(nu))
    plt.plot(time, x)
    plt.xlabel('time')
    plt.ylabel('x')
plt.show()
times

plt.title('T = '+str(T)+', nu = '+str(nu))
plt.xlabel('time')
plt.ylabel('u(t)')
plt.plot(time[:n-1], control[:n-1])
plt.show()


