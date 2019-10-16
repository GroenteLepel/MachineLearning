#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:39:07 2019

@author: laurens
"""
import os
import numpy as np
import math as m
import pandas as pd
import seaborn as sns
from sympy.stats import *
import matplotlib.pyplot as plt
os.chdir('/Users/laurens/Programmeren/CDS: Machine learning')
data=pd.read_csv('datayag.csv', sep=';', engine='python')

data.describe()
data.isnull().sum() #Check if there are missing values
data.duplicated().sum() #Check if there are duplicate values

Xraw=data.values[:,0]
Yraw=data.values[:,1]
X=(Xraw-np.mean(Xraw))/(np.std(Xraw))
Y=(Yraw-np.mean(Yraw))/(np.std(Yraw))

plt.plot(X,Y,'o')
plot.show()
XY=np.zeros(len(X))
for i in range(0,len(X)):
    XY[i]=1
    

mu1=[-1,1]
mu2=[1,-1]
S1=np.array([[1,0],[0,1]])
S2=np.array([[1,0],[0,1]])
pi1=0.5
pi2=0.5
r1=np.zeros(len(X))
r2=np.zeros(len(X))
from scipy.stats import multivariate_normal
for i in range(0,len(X)):
    r1[i]=pi1*multivariate_normal.pdf((X[i],Y[i]), mean=mu1, cov=S1)/(pi1*multivariate_normal.pdf((X[i],Y[i]), mean=mu1, cov=S1)+pi2*multivariate_normal.pdf((X[i],Y[i]), mean=mu2, cov=S2))
    r2[i]=pi2*multivariate_normal.pdf((X[i],Y[i]), mean=mu2, cov=S2)/(pi1*multivariate_normal.pdf((X[i],Y[i]), mean=mu1, cov=S1)+pi2*multivariate_normal.pdf((X[i],Y[i]), mean=mu2, cov=S2))