#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:36:09 2020

@author: Ludo van Alst
"""

import numpy as np

n_neurons = 160

def get_clamped_expectations_from_file(readfilepath: str, writefilepath: str):
    expectation_vector_c = np.zeros(n_neurons)
    expectation_matrix_c = np.zeros((n_neurons,n_neurons))
    
    readfilepath_2 = readfilepath[:-4] + '2.txt'
    with open(readfilepath, 'r') as f:
        # get the amount of samples and reset pointer to start of file
        line = f.readline()
        n_samples = len(line.split())
        f.seek(0)
        
        with open(readfilepath_2, 'r') as g:
            for index_line1, line1 in enumerate(f):
                print('line {0:d}'.format(index_line1+1))
                line1split = line1.split()
                for index_line2, line2 in enumerate(g):
                    if index_line2 >= index_line1:
                        line2split = line2.split()
                        for index_elt1, elt1 in enumerate(line1split):
                            if index_line2 == index_line1:
                                expectation_vector_c[index_line1] += (int(elt1)-0.5)/0.5
                            expectation_matrix_c[index_line1][index_line2] += (int(elt1)-0.5)/0.5 * (int(line2split[index_elt1])-0.5)/0.5
                            if index_line2 != index_line1:
                                expectation_matrix_c[index_line2][index_line1] += (int(elt1)-0.5)/0.5 * (int(line2split[index_elt1])-0.5)/0.5
                g.seek(0)
    
    expectation_vector_c = expectation_vector_c / n_samples
    expectation_matrix_c = expectation_matrix_c / n_samples
    
    with open(writefilepath, 'w') as f:
        for elt in expectation_vector_c:
            f.write('{0:f} '.format(elt))
        f.write('\n\n')
        
        for line in expectation_matrix_c:
            for elt in line:
                f.write('{0:f} '.format(elt))
            f.write('\n')      
              
    return expectation_vector_c, expectation_matrix_c