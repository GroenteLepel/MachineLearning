#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:36:09 2020

@author: Ludo van Alst
"""

import numpy as np

n_neurons = 160

def reformat_clamped_expectations_file(readfilepath: str, writefilepath: str):
    expectation_vector_c = np.zeros(n_spins)
    expectation_matrix_c = np.zeros((n_spins,n_spins))
    
    with open(readfilepath, 'r') as f:
        firstline = f.readline()
        for index, elt in enumerate(firstline.split()):
            expectation_vector_c[index] = float(elt)
        f.readline()
        for index1, line in enumerate(f):
            for index2, elt in enumerate(line.split()):
                expectation_matrix_c[index1][index2] = float(elt)
    
    with open(writefilepath, 'w') as f:
        for elt in expectation_vector_c:
            f.write('{0:f} '.format(elt))
        f.write('\n\n')
        
        for lineindex, line in enumerate(expectation_matrix_c):
            for eltindex, elt in enumerate(line):
                if lineindex == eltindex:
                    f.write('{0:f} '.format(0))
                else: 
                    f.write('{0:f} '.format(elt))
            f.write('\n')      
              
    return expectation_vector_c, expectation_matrix_c