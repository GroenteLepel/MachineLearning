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
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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



mu1=[-1,1]
mu2=[1,-1]
S1=np.array([[1,0],[0,1]])
S2=np.array([[1,0],[0,1]])
pi1=0.5
pi2=0.5
r1=np.zeros(len(X))
r2=np.zeros(len(X))
from scipy.stats import multivariate_normal
#%% 
mu1=[-1,1]
mu2=[1,-1]
S1=np.array([[1,0],[0,1]])
S2=np.array([[1,0],[0,1]])
pi1=0.5
pi2=0.5
r1=np.zeros(len(X))
r2=np.zeros(len(X))
Niterations=1
for j in range(0,Niterations):
    Snew1=np.array([[0,0],[0,0]])
    Snew2=np.array([[0,0],[0,0]])
    for i in range(0,len(X)):
        r1[i]=pi1*multivariate_normal.pdf((X[i],Y[i]), mean=mu1, cov=S1)/(pi1*multivariate_normal.pdf((X[i],Y[i]), mean=mu1, cov=S1)+pi2*multivariate_normal.pdf((X[i],Y[i]), mean=mu2, cov=S2))
        r2[i]=pi2*multivariate_normal.pdf((X[i],Y[i]), mean=mu2, cov=S2)/(pi1*multivariate_normal.pdf((X[i],Y[i]), mean=mu1, cov=S1)+pi2*multivariate_normal.pdf((X[i],Y[i]), mean=mu2, cov=S2))
        Snew1=Snew1+r1[i]*np.array([[X[i]-mu1[0]],[Y[i]-mu1[1]]])*np.array([X[i]-mu1[0],Y[i]-mu1[1]])
        Snew2=Snew2+r2[i]*np.array([[X[i]-mu2[0]],[Y[i]-mu2[1]]])*np.array([X[i]-mu2[0],Y[i]-mu2[1]])
    mu1[0]=np.sum(r1*X)/np.sum(r1)
    mu1[1]=np.sum(r1*Y)/np.sum(r1)
    mu2[0]=np.sum(r2*X)/np.sum(r2)
    mu2[1]=np.sum(r2*Y)/np.sum(r2)
    S1=Snew1/np.sum(r1)
    S2=Snew2/np.sum(r1)
    pi1=np.mean(r1)
    pi2=np.mean(r2)
    print('step',j+1)
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='auto')
    w,v=np.linalg.eig(S1)
    w2,v2=np.linalg.eig(S2)
    index = np.where(w == np.max(w))
    index2 = np.where(w2 == np.max(w2))
    ori=np.arctan(v[1][index]/v[0][index])
    ori2=np.arctan(v2[1][index]/v2[0][index])
    major,minor=2*np.sqrt(5.991*w[0]),2*np.sqrt(5.991*w[1])
    major2,minor2=2*np.sqrt(5.991*w2[0]),2*np.sqrt(5.991*w2[1])
    ellipse = Ellipse(mu1,minor,major,angle=ori,alpha=0.1,color='r')
    ellipse2 = Ellipse(mu2,minor2,major2,angle=ori2,alpha=0.1,color='b')
    ax.add_patch(ellipse)
    ax.add_patch(ellipse2)
    ax.scatter(X,Y,cmap='seismic',c=r1)
    fig.show()



#%%
fig = plt.figure()
ax = fig.add_subplot(111, aspect='auto')
w,v=np.linalg.eig(S1)
w2,v2=np.linalg.eig(S2)
index = np.where(w == np.max(w))
index2 = np.where(w2 == np.max(w2))
ori=np.arctan(v[1][index]/v[0][index])
ori2=np.arctan(v2[1][index]/v2[0][index])
major,minor=2*np.sqrt(5.991*w[0]),2*np.sqrt(5.991*w[1])
major2,minor2=2*np.sqrt(5.991*w2[0]),2*np.sqrt(5.991*w2[1])
ellipse = Ellipse(mu1,minor,major,angle=ori,alpha=0.1,color='r')
ellipse2 = Ellipse(mu2,minor2,major2,angle=ori2,alpha=0.1,color='b')
ax.add_patch(ellipse)
ax.add_patch(ellipse2)
ax.scatter(X,Y,cmap='seismic',c=r1)
fig.show()