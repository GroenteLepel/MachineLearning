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
train_coords_raw = data['mnist']['train_images'][0][0] / 255
test_coords_raw = data['mnist']['test_images'][0][0] / 255
train_labels_raw = data['mnist']['train_labels'][0][0]
test_labels_raw = data['mnist']['test_labels'][0][0]

# Selecting data, only 3's and 7's needed
indextrain = [i for i, x in enumerate(train_labels_raw) if x == 3 or x == 7]
indextest = [i for i, x in enumerate(test_labels_raw) if x == 3 or x == 7]

# *train_new indicates the array but with only the 3's and 7s', labeled as
# 0's and 1's respectively.
train_coords = np.zeros((28, 28, len(indextrain)))
test_coords = np.zeros((28, 28, len(indextest)))
train_labels = np.zeros(len(indextrain))
test_labels = np.zeros(len(indextest))

# manipulating data to boolean values instead of 3 or 7
# TODO: only every label that needs to be 0 can be changed, since Ttrain_new
#  already is an array of zeros.
# TODO: remove the damned for-loop.

for i, x in enumerate(indextrain):
    if train_labels_raw[x] == 3:
        train_labels[i] = 0
    else:
        train_labels[i] = 1
    train_coords[:, :, i] = train_coords_raw[:, :, x]

for i, x in enumerate(indextest):
    if test_labels_raw[x] == 3:
        test_labels[i] = 0  # The 3's get label 0
    else:
        test_labels[i] = 1  # The 7's get label 1
    test_coords[:, :, i] = test_coords_raw[:, :, x]

# reshaping 28x28 matrices to one long array of 784 entries
# each row now is a data point and each column the dimension
train_coords = np.transpose(np.reshape(train_coords, (784, len(train_labels))))
test_coords = np.transpose(np.reshape(test_coords, (784, len(test_labels))))
train_labels = train_labels.reshape(len(train_labels), 1)
test_labels = test_labels.reshape(len(test_labels), 1)

# %% Grad descent for different etas

# declaring amount of steps and indicators for all the values.
n_steps = 400
values = np.linspace(1, n_steps, n_steps)

eta = 0.3
train_loss_03, test_loss_03 = gradient_descent(train_coords, train_labels,
                                               test_coords, test_labels,
                                               eta, epochs=n_steps)
# We find that the Testloss had a minimum value around 6500 epochs. Going to
# 10.000 results in overfitting. Thus for this eta=0.3 we would suggest using
# around 6500 epochs. Now we look at what happens when we use different eta


eta = 0.9
train_loss_09, test_loss_09 = gradient_descent(train_coords, train_labels,
                                               test_coords, test_labels,
                                               eta, epochs=n_steps)

eta = 0.1
train_loss_01, test_loss_01 = gradient_descent(train_coords, train_labels,
                                               test_coords, test_labels,
                                               eta, epochs=n_steps)


# %% Momentum
eta = 0.5
alpha = 0.5

train_loss_04, test_loss_04 = gradient_descent(train_coords, train_labels,
                                               test_coords, test_labels,
                                               eta, momentum_step=alpha,
                                               epochs=n_steps)


# %% Weight decay
eta = 0.5
alpha = 0.5
lab = 0.1

train_loss_05, test_loss_05 = gradient_descent(train_coords, train_labels,
                                               test_coords, test_labels,
                                               eta, momentum_step=alpha,
                                               decay_factor=lab, epochs=n_steps)


# TODO: write away this data into a text file and plot using a different python
#  script.
plt.plot(values, train_loss_03, label='Etraining03')
plt.plot(values, test_loss_03, label='Etest03')

plt.plot(values, train_loss_09, label='Etraining09')
plt.plot(values, test_loss_09, label='Etest09')

plt.plot(values, train_loss_01, label='Etraining01')
plt.plot(values, test_loss_01, label='Etest01')

plt.plot(values, train_loss_04, label='Etraining04 (mom)')
plt.plot(values, test_loss_04, label='Etest04 (mom)')

plt.plot(values, train_loss_05, label='Etraining05 (mom, weighted)')
plt.plot(values, test_loss_05, label='Etest05 (mom, weighted)')

plt.title('Entropy versus epochs for grad descent')
plt.xlabel('epochs')
plt.ylabel('entropy')
plt.legend()
plt.savefig('grad_descent.png')
plt.show()

# %% Extras
## calculate time elapsed for one grad calculation
# time_start = time.clock()
# grad(w,Xtrain_new,Ttrain_new) #Testing if it works #Testing if it works
# print((time.clock() - time_start))
#
## find minimum of testloss
# print(int(np.where(Testloss==min(Testloss))[0]))
