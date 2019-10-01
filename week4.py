#!/usr/bin/env python
# coding: utf-8

'''
CDS Machine Learning week 4
Radboud University
authors: Ludo van Alst, Laurens Sluijterman, Daniël Kok
'''

# %% Importing modules
import scipy.io
import numpy as np
import time
# get_ipython().run_line_magic('matplotlib', 'notebook') #voor Laurens de jupiter-banaan
import matplotlib.pyplot as plt
import os
from ml_functions import *

# pick your path
os.chdir('C:/Users/Daniël/iCloudDrive/Documents/CDSMachineLearning')

# %% Importing data
data = scipy.io.loadmat('mnistAll.mat')

# X indicates the coordinates of the images, normalized between 0-1 by dividing
# by 255.
# T indicates the label of each image, which we cherry-pick only the 3's and 7's
# and give these a 0 or 1 according to the 3 or 7 later on.
Xtrain = data['mnist']['train_images'][0][0] / 255
Xtest = data['mnist']['test_images'][0][0] / 255
Ttrain = data['mnist']['train_labels'][0][0]
Ttest = data['mnist']['test_labels'][0][0]

# Selecting data, only 3's and 7's needed
indextrain = [i for i, x in enumerate(Ttrain) if x == 3 or x == 7]
indextest = [i for i, x in enumerate(Ttest) if x == 3 or x == 7]

# *train_new indicates the array but with only the 3's and 7s', labeled as
# 0's and 1's respectively.
Xtrain_new = np.zeros((28, 28, len(indextrain)))
Xtest_new = np.zeros((28, 28, len(indextest)))
Ttrain_new = np.zeros(len(indextrain))
Ttest_new = np.zeros(len(indextest))

# manipulating data to boolean values instead of 3 or 7
# TODO: only every label that needs to be 0 can be changed, since Ttrain_new
#  already is an array of zeros.
# TODO: remove the damned for-loop.

for i, x in enumerate(indextrain):
    if Ttrain[x] == 3:
        Ttrain_new[i] = 0
    else:
        Ttrain_new[i] = 1
    Xtrain_new[:, :, i] = Xtrain[:, :, x]

for i, x in enumerate(indextest):
    if Ttest[x] == 3:
        Ttest_new[i] = 0  # The 3's get label 0
    else:
        Ttest_new[i] = 1  # The 7's get label 1
    Xtest_new[:, :, i] = Xtest[:, :, x]

# reshaping 28x28 matrices to one long array of 784 entries
# each row now is a datapoint and each column the dimension
Xtrain_new = np.transpose(np.reshape(Xtrain_new, (784, len(Ttrain_new))))
Xtest_new = np.transpose(np.reshape(Xtest_new, (784, len(Ttest_new))))
Ttrain_new = Ttrain_new.reshape(len(Ttrain_new), 1)
Ttest_new = Ttest_new.reshape(len(Ttest_new), 1)


# %% Grad descent for different etas
# plotting entropy as function of epochs
epochs = 4000
values = np.linspace(1, epochs, epochs)

eta = 0.3
time_start = time.clock()
Trainloss03, Testloss03 = entropy(Xtrain_new, Ttrain_new, Xtest_new, Ttest_new,
                                  0.3, epochs)
print('eta=0.3 done in {0:d} seconds'.format(int(time.clock() - time_start)))
plt.plot(values, Trainloss03, label='Etraining03')
plt.plot(values, Testloss03, label='Etest03')
# We find that the Testloss had a minimum value around 6500 epochs. Going to 
# 10.000 results in overfitting. Thus for this eta=0.3 we would suggest using
# around 6500 epochs. Now we look at what happens when we use different eta

eta = 0.9
time_start = time.clock()
Trainloss09, Testloss09 = entropy(Xtrain_new, Ttrain_new, Xtest_new, Ttest_new,
                                  0.9, epochs)
print('eta=0.9 done in {0:d} seconds'.format(int(time.clock() - time_start)))
plt.plot(values, Trainloss09, label='Etraining09')
plt.plot(values, Testloss09, label='Etest09')

eta = 0.1
time_start = time.clock()
Trainloss01, Testloss01 = entropy(Xtrain_new, Ttrain_new, Xtest_new, Ttest_new,
                                  0.1, epochs)
print('eta=0.1 done in {0:d} seconds'.format(int(time.clock() - time_start)))
plt.plot(values, Trainloss01, label='Etraining01')
plt.plot(values, Testloss01, label='Etest01')

plt.title('Entropy versus epochs for grad descent')
plt.xlabel('epochs')
plt.ylabel('entropy')
plt.legend()
plt.savefig('grad_descent.png')
plt.show()

# %% Momentum
eta = 0.5
alpha = 0.5
epochs = 4000
values = np.linspace(1, epochs, epochs)
time_start = time.clock()
Trainloss04, Testloss04 = momentum(Xtrain_new, Ttrain_new, Xtest_new, Ttest_new,
                                   eta, alpha, epochs)
print('eta=0.7 done in {0:d} seconds'.format(int(time.clock() - time_start)))
plt.plot(values, Trainloss04, label='Etraining04')
plt.plot(values, Testloss04, label='Etest04')

# %% Weight decay
eta = 0.5
alpha = 0.5
lab = 0.1
epochs = 4000
values = np.linspace(1, epochs, epochs)
time_start = time.clock()
Trainloss05, Testloss05 = weightdecay(Xtrain_new, Ttrain_new, Xtest_new,
                                      Ttest_new, eta, alpha, lab, epochs)
print('eta=0.7 done in {0:d} seconds'.format(int(time.clock() - time_start)))
plt.plot(values, Trainloss04, label='Etraining04')
plt.plot(values, Testloss04, label='Etest04')

# %% Extras
## calculate time elapsed for one grad calculation
# time_start = time.clock()
# grad(w,Xtrain_new,Ttrain_new) #Testing if it works #Testing if it works
# print((time.clock() - time_start))
#
## find minimum of testloss
# print(int(np.where(Testloss==min(Testloss))[0]))
