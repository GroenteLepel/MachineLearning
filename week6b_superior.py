import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

# os.chdir('C:/Users/Daniël/iCloudDrive/Documents/CDSMachineLearning')
os.chdir('/Users/daniel/Documents/CDSMachineLearning')


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


def initialize_em():
    means = np.array([[-1., 1.],  # k = 1

                      [1., -1.]])  # k = 2

    covariance_mat = np.array([[[1., 0.],  # k = 1
                                [0., 1.]],

                               [[1., 0.],  # k = 2
                                [0., 1.]]])
    covariance_mat *= 0.05

    pi = np.array([0.5,  # k = 1

                   0.5])  # k = 2

    return means, covariance_mat, pi


def e_step(data, pi_values, means, covariances):
    """
    Evaluate the responsibilities using the current parameter values (i. e.
    updates the colors of the data points to coincide with the overlayed
    gaussians).
    :param data: X and Y coordinates of data points to be clustered.
    :param pi_values:
    :param means:
    :param covariances:
    :return: updated responsibility values in an array.
    """
    # TODO: find a better and more elegant way to do this.
    numerator_1 = \
        pi_values[0] * multivariate_normal.pdf(np.transpose(data),
                                               mean=means[0],
                                               cov=covariances[0])
    numerator_2 = \
        pi_values[1] * multivariate_normal.pdf(np.transpose(data),
                                               mean=means[1],
                                               cov=covariances[1])
    denominator = \
        numerator_1 + numerator_2

    resp_1 = numerator_1 / denominator
    resp_2 = numerator_2 / denominator

    if (resp_1 + resp_2).any() != 1:
        print("!!probabilities do not sum up to one.!!")

    return np.array([resp_1, resp_2])


def m_step(data, resp):
    """
    Re-estimates the parameters using the current responsibilities (i. e.
    updates the gaussians).
    :param data: X and Y coordinates of data points to be clustered.
    :param resp: array of responsibilities for each data point.
    :return: Updated mu, covariance matrix and pi values.
    """
    # Sum up all the responsibility values for each gaussian k.
    n_responsible = resp.sum(axis=1)

    # Initialise the mu, cov and pi which are to be updated.
    mu_new, covariance_new, pi_new = initialize_em()

    # TODO: these for loops can be change to a matrix multiplication of some
    #  sort.
    for i in range(len(mu_new)):
        for j in range(len(mu_new[0])):
            mu_new[i][j] = resp[i].dot(data[j]) / n_responsible[i]

    for i in range(len(covariance_new)):
        covariance_new[i] = (resp * (data - mu_new[i][:, None])) \
                                .dot(np.transpose(data - mu_new[i][:, None])) \
                            / n_responsible[i]

    for i in range(len(pi_new)):
        pi_new[i] = n_responsible[i] / len(resp)

    return mu_new, covariance_new, pi_new


def plot_faithful(data, resp_data, means, covariance):
    fig, ax = plt.subplots(1, 1)

    edgecolors = ['red', 'blue']

    for i in range(len(resp_data)):
        lambda_, v = np.linalg.eig(covariance[i])
        lambda_ = np.sqrt(lambda_)

        ell = Ellipse(xy=means[i],
                      width=lambda_[0] / np.sqrt(2), height=lambda_[1],
                      angle=np.rad2deg(np.arccos(v[0, 0])),
                      facecolor='none',
                      edgecolor=edgecolors[i], linewidth=3)

        ax.add_artist(ell)

    ax.scatter(data[0], data[1], c=resp_data[0], cmap='bwr', vmin=0, vmax=1)

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])

    fig.show()


# %% Main

# Read data.
faithful_data = read_faithful()

# Initialize the gaussians.
mu, cov, pi = initialize_em()
# Plot the gaussians and data to show how it is initialized.
plot_faithful(faithful_data, np.zeros((2, len(faithful_data[0]))), mu, cov)
# Calculate the responsibilities for the gaussians belonging to each data point.
r = e_step(faithful_data, pi, mu, cov)
# Show this in a plot.
plot_faithful(faithful_data, r, mu, cov)

# Update in loops
n_steps = 50
n_plots = 6
for i in range(n_steps):
    mu, cov, pi = m_step(faithful_data, r)
    r = e_step(faithful_data, pi, mu, cov)

    if i != 0 and i % int(n_steps / n_plots) == 0:
        print("plot at step {0:d}".format(i))
        plot_faithful(faithful_data, r, mu, cov)
