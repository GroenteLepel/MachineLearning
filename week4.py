#!/usr/bin/env python
# coding: utf-8

'''
CDS Machine Learning week 4
Radboud University
authors: Ludo van Alst, Laurens Sluijterman, Daniël Kok
'''

# %% Importing modules
import scipy.io
import matplotlib.pyplot as plt
import os
from ml_functions import *
import csv

# pick your path
#os.chdir('C:/Users/Daniël/iCloudDrive/Documents/CDSMachineLearning')
# os.chdir('C:/Users/Daniël/iCloudDrive/Documents/CDSMachineLearning')
os.chdir('/Users/daniel/Documents/CDSMachineLearning')
# os.chdir('/home/lvalst/Courses/Machine Learning/week4')
os.chdir('/Users/laurens/Programmeren/CDS: Machine learning/MachineLearning')


# %% Importing data
data = scipy.io.loadmat('mnistAll.mat')

# X indicates the coordinates of the images, normalized between 0-1 by dividing
# by 255.
# T indicates the label of each image, which we cherry-pick only the 3's and 7's
# and give these a 0 or 1 according to the 3 or 7 later on.
train_coords_raw = data['mnist']['train_images'][0][0] / MAX_INT
test_coords_raw = data['mnist']['test_images'][0][0] / MAX_INT
train_labels_raw = data['mnist']['train_labels'][0][0]
test_labels_raw = data['mnist']['test_labels'][0][0]

# Selecting data, only 3's and 7's needed
indextrain = [i for i, x in enumerate(train_labels_raw) if x == 3 or x == 7]
indextest = [i for i, x in enumerate(test_labels_raw) if x == 3 or x == 7]

# *train_new indicates the array but with only the 3's and 7s', labeled as
# 0's and 1's respectively.
train_coords = np.zeros((RESOLUTION, RESOLUTION, len(indextrain)))
test_coords = np.zeros((RESOLUTION, RESOLUTION, len(indextest)))
train_labels = np.zeros(len(indextrain))
test_labels = np.zeros(len(indextest))

# manipulating data to boolean values instead of 3 or 7
# TODO: only every label that needs to be 0 can be changed, since Ttrain_new
#  already is an array of zeros.
# TODO: remove the for-loop.

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
train_coords = np.transpose(np.reshape(train_coords,
                                       (RES_SQ, len(train_labels))))
test_coords = np.transpose(np.reshape(test_coords,
                                      (RES_SQ, len(test_labels))))
train_labels = train_labels.reshape(len(train_labels), 1)
test_labels = test_labels.reshape(len(test_labels), 1)

del train_coords_raw, test_coords_raw, train_labels_raw, test_labels_raw
del indextest, indextrain

# put a 1 in front for each datapoint (this is the x_0 coordinate)
train_coords = np.insert(train_coords, 0, 1, axis=1)
test_coords = np.insert(test_coords, 0, 1, axis=1)

# %% Grad descent for different etas

# declaring amount of steps and indicators for all the values.
n_steps=10000
values = np.linspace(1, n_steps, n_steps)
w = np.random.normal(0, 1. / 10, (RES_SQ+1, 1))

# Hdiag, Hmat = hessian(w, train_coords, 0.1)
# np.linalg.inv(Hmat)

eta = 0.3
lab = 0.1

trainl, testl, w3 = gradient_descent(train_coords, train_labels, test_coords,
                                    test_labels, congrad_descent=True,
                                    epochs=n_steps)

#
eta = 0.3
train_loss_03, test_loss_03,w3 = gradient_descent(train_coords, train_labels,
                                               test_coords, test_labels,
                                               step_strength=eta,
                                               epochs=n_steps,momentum_step=0.5
                                               ,decay_factor=0.1)
# # We find that the Testloss had a minimum value around 6500 epochs. Going to
# # 10.000 results in overfitting. Thus for this eta=0.3 we would suggest using
# # around 6500 epochs. Now we look at what happens when we use different eta
#
#
eta = 0.9
train_loss_09, test_loss_09,w9 = gradient_descent(train_coords, train_labels,
                                               test_coords, test_labels,
                                               step_strength=eta,
                                               epochs=n_steps)
#
eta = 0.1
train_loss_01, test_loss_01,w1 = gradient_descent(train_coords, train_labels,
                                               test_coords, test_labels,
                                               step_strength=eta,
                                               epochs=n_steps)
#
#
# # %% Momentum
# eta = 0.5
# alpha = 0.5
#
# train_loss_04, test_loss_04 = gradient_descent(train_coords, train_labels,
#                                                test_coords, test_labels,
#                                                step_strength=eta,
#                                                momentum_step=alpha,
#                                                epochs=n_steps)
#
#
# # %% Weight decay
# eta = 0.5
# alpha = 0.5
# lab = 0.1
#
# train_loss_05, test_loss_05 = gradient_descent(train_coords, train_labels,
#                                                test_coords, test_labels,
#                                                step_strength=eta,
#                                                momentum_step=alpha,
#                                                decay_factor=lab, epochs=n_steps)

# %% Newtonian decay
# eta = 0.3
# train_loss_03, test_loss_03, weights = gradient_descent(train_coords, train_labels,
#                                                test_coords, test_labels,
#                                                eta, epochs=n_steps)

# eta = 0.3
# train_loss_03, test_loss_03, weights_03 = gradient_descent(train_coords,
#                                                            train_labels,
#                                                            test_coords,
#                                                            test_labels,
#                                                            eta, epochs=n_steps)
# We find that the Testloss had a minimum value around 6500 epochs. Going to
# 10.000 results in overfitting. Thus for this eta=0.3 we would suggest using
# around 6500 epochs. Now we look at what happens when we use different eta


# eta = 0.9
# train_loss_09, test_loss_09 = gradient_descent(train_coords, train_labels,
#                                                test_coords, test_labels,
#                                                eta, epochs=n_steps)
#
# eta = 0.1
# train_loss_01, test_loss_01 = gradient_descent(train_coords, train_labels,
#                                                test_coords, test_labels,
#                                                eta, epochs=n_steps)
# eta = 0.9
# train_loss_09, test_loss_09, weights_09 = gradient_descent(train_coords,
#                                                            train_labels,
#                                                            test_coords,
#                                                            test_labels,
#                                                            eta, epochs=n_steps)
#
# eta = 0.1
# train_loss_01, test_loss_01, weights_01 = gradient_descent(train_coords,
#                                                            train_labels,
#                                                            test_coords,
#                                                            test_labels,
#                                                            eta, epochs=n_steps)

# %% Newtonian
# eta = 0.5
# alpha = 0.5
# #
# train_loss_04, test_loss_04, w = gradient_descent(train_coords, train_labels,
#                                                   test_coords, test_labels,
#                                                   eta, momentum_step=alpha,
#                                                   epochs=n_steps)
#
# train_loss_newt, test_loss_newt, w = gradient_descent(train_coords,
#                                                       train_labels,
#                                                       test_coords,
#                                                       test_labels,
#                                                       eta,
#                                                       newtonian=True,
#                                                       epochs=n_steps)
#
# print(np.min(train_loss_newt))
# print(np.min(test_loss_newt))
# train_loss_04, test_loss_04 = gradient_descent(train_coords, train_labels,
#                                                test_coords, test_labels,
#                                                eta, momentum_step=alpha,
#                                                epochs=n_steps)

# print(np.min(train_loss_newt))
# print(np.min#
# train_loss_04, test_loss_04, weights_04 = gradient_descent(train_coords,
#                                                            train_labels,
#                                                            test_coords,
#                                                            test_labels,
#                                                            eta,
#                                                            momentum_step=alpha,
#                                                            epochs=n_steps)
#
# train_loss_newt, test_loss_newt, w = gradient_descent(train_coords, train_labels
#                                                       , test_coords,
#                                                       test_labels,
#                                                       newtonian=True,
#                                                       epochs=n_steps)
# (test_loss_newt))
# train_loss_05, test_loss_05, weights_05 = gradient_descent(train_coords,
#                                                            train_labels,
#                                                            test_coords,
#                                                            test_labels,
#                                                            eta,
#                                                            momentum_step=alpha,
#                                                            decay_factor=lab,
#                                                            epochs=n_steps)

# After 10 iterations, Etrain = 0.10, Etest = 0.14, so way to high. something
# is wrong...
=======
lab = 0.1

# %%Newtonian
n_steps = 100
values = np.linspace(1, n_steps, n_steps)
trainl_n, testl_n, w_n = gradient_descent(train_coords, train_labels,
                                          test_coords,
                                          test_labels, decay_factor=lab,
                                          newtonian=True, epochs=n_steps)

train_n_class = classification_check(train_coords, train_labels, w_n)
test_n_class = classification_check(test_coords, test_labels, w_n)

# %% Plotting
# TODO: write away this data into a text file and plot using a different python
#  script.
plt.plot(values, train_loss_03, label='Etraining03')
plt.plot(values, test_loss_03, label='Etest03')
##
#plt.plot(values, train_loss_09, label='Etraining09')
#plt.plot(values, test_loss_09, label='Etest09')
#
# plt.plot(values, train_loss_01, label='Etraining01')
# plt.plot(values, test_loss_01, label='Etest01')
#
# plt.plot(values, train_loss_04, label='Etraining04 (mom)')
# plt.plot(values, test_loss_04, label='Etest04 (mom)')
#
# plt.plot(values, train_loss_05, label='Etraining05 (mom, weighted)')
# plt.plot(values, test_loss_05, label='Etest05 (mom, weighted)')


#plt.plot(values, trainl, label='Etraining (newt)')
#plt.plot(values, testl, label='Etest (newt)')

# # plt.yscale('log')
# # plt.xscale('log')
#
plt.title('Entropy versus epochs for grad descent')
plt.xlabel('epochs')

plt.plot(values, trainl_n, label=r'$E_{train}$')
plt.plot(values, testl_n, label=r'$E_{test}$')

plt.title('Newtonian Gradient Descent')
plt.xlabel('iterations')
plt.ylabel('entropy')
plt.legend()
plt.savefig('newtonian.png')
# plt.show()
plt.clf()

# %% Line Search

n_steps = 224
values = np.linspace(1, n_steps, n_steps)
trainl_ls, testl_ls, w_ls = gradient_descent(train_coords, train_labels,
                                             test_coords, test_labels,
                                             decay_factor=lab, linesearch=True,
                                             epochs=n_steps)

train_ls_class = classification_check(train_coords, train_labels, w_ls)
test_ls_class = classification_check(test_coords, test_labels, w_ls)

plt.plot(values, trainl_ls, label=r'$E_{train}$')
plt.plot(values, testl_ls, label=r'$E_{test}$')

plt.title('Line Search')
plt.xlabel('iterations')
plt.ylabel('entropy')
plt.legend()
plt.savefig('linesearch.png')
# plt.show()
plt.clf()

# %% Conjugate

n_steps = 106
values = np.linspace(1, n_steps, n_steps)
trainl_cd, testl_cd, w_cd = gradient_descent(train_coords, train_labels,
                                             test_coords, test_labels,
                                             decay_factor=lab,
                                             congrad_descent=True,
                                             epochs=n_steps)

train_cd_class = classification_check(train_coords, train_labels, w_cd)
test_cd_class = classification_check(test_coords, test_labels, w_cd)

# %%
plt.plot(values, trainl_cd, label=r'$E_{train}$')
plt.plot(values, testl_cd, label=r'$E_{test}$')

plt.title('Conjugate Gradient Descent')
plt.xlabel('iterations')
plt.ylabel('entropy')
plt.legend()
plt.savefig('congraddescent.png')
# plt.show()
plt.clf()

# %% Stochastic Gradient Descent
n_steps = 10000
values = np.linspace(1, n_steps, n_steps)
trainl_st, testl_st, w_st = gradient_descent(train_coords, train_labels,
                                             test_coords, test_labels,
                                             decay_factor=lab,
                                             batch_size=1000,
                                             epochs=n_steps)

train_st_class = classification_check(train_coords, train_labels, w_st)
test_st_class = classification_check(test_coords, test_labels, w_st)

# %%
plt.plot(values, trainl_st, label=r'$E_{train}$')
plt.plot(values, testl_st, label=r'$E_{test}$')

plt.title(r'Stochastic Gradient Descent ($N = 1000$')
plt.xlabel('iterations')
plt.ylabel('entropy')
plt.legend()
plt.savefig('stochastic.png')
# plt.show()
plt.clf()


