#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:36:09 2020

@author: Ludo van Alst
"""

import numpy as np

n_neurons = 160
batchsize = 5000
n_measurements = 283041

def get_observed_rates(readfilepath: str, writefilepath: str = None):
    observed_rates = np.zeros(2**10)
    
    batch_start = 0
    with open(readfilepath, 'r') as f:
        stop = False
        while not stop:
            batch_states = [''] * batchsize
            for _ in range(10):
                line = f.readline()
                linesplit = line.split()
                if len(linesplit) - batch_start <= batchsize:
                    stop = True
                    linesplit_batch = linesplit[batch_start:]
                else:
                    linesplit_batch = linesplit[batch_start:batch_start+batchsize]
                
                for i, elt in enumerate(linesplit_batch):
                    batch_states[i] += elt
            
            for batch_state in batch_states:
                if batch_state == '':
                    continue
                int_repr = int(batch_state, 2)  # the string "batch_state"  is in base 2
                observed_rates[int_repr] += 1
            
            batch_start += batchsize
            f.seek(0)
            
    if writefilepath:
        with open(writefilepath, 'w') as g:
            for state_binary, amount in enumerate(observed_rates):
                bin_repr_string = bin(state_binary)[2:]
                g.write(f'{int(bin_repr_string):010}' + '\t\t\t{0:d}\n'.format(int(amount)))
                
    observed_rates = np.multiply(np.array(observed_rates), 50.0/n_measurements)
              
    return observed_rates