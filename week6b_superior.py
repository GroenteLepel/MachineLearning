import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

os.chdir('C:/Users/Daniël/iCloudDrive/Documents/CDSMachineLearning')


# %% Functions


def read_faithful():
    """
    Read the faithful.dat file
    :return: the normalised x and y data as one array
    """
    # Creates an array of 3, 272 with the first element is the index, second the
    # 'x' coordinate, third the 'y'
    data_raw = np.loadtxt("faithful.dat", skiprows=26, unpack=True)

    # calculate the mean and std of the three elements
    data = (data_raw - data_raw.mean(axis=1)[:, None]) \
           / data_raw.std(axis=1)[:, None]

    return data[-2:]


def initialize_em(x_data):
    means = np.array([[-1, 1], [1, -1]])
    pi = np.array([0.5, 0.5])
    covariance_mat = np.array([[[1, 0], [0, 1]],
                                 [[1, 0], [0, 1]]])
    responsibilities = np.zeros((2, len(x_data)))

    return means, pi, covariance_mat, responsibilities


def e_step(x_data, y_data, weights, mean_data, cov_data):
    numerator = \
        weights[0] * multivariate_normal.pdf((x_data, y_data),
                                             mean=mean_data[0],
                                             cov=cov_data[0])
    denominator = \
        weights[0] * multivariate_normal.pdf((x_data, y_data),
                                             mean=mean_data[0],
                                             cov=cov_data[0]) \
        + weights[1] * multivariate_normal.pdf((x_data, y_data),
                                               mean=mean_data[1],
                                               cov=cov_data[1])

def plot_faithful(x_data, y_data):
    fig, ax = plt.subplots(1, 1)

    ax.scatter(x_data, y_data)

    fig.show()


# %% Main

x, y = read_faithful()

plot_faithful(x, y)

mu, pi, cov, r = initialize_em(x)

iterations = 1
# for i in range(iterations):
#     cov_new = np.zeros(np.shape(cov))
#
#     for j in range(len(x)):
