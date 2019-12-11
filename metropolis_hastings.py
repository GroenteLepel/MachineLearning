import numpy as np
from scipy.stats import multivariate_normal
from p_star_distribution import objective_function


def normal(x, mean=0):
    """
    :return: float, probability that x is in a 3D-gaussian with identity matrix
    as covariance.
    """

    return np.exp(-1 * np.dot((x - mean), (x - mean)) / 2) / \
           (np.sqrt(2 * np.pi) ** 3)


def metropolis_hastings(p, data, labels, n_points=1000, n_dims=3):
    """
    Method for sampling data from a distribution p
    :param p: distribution to sample using this method
    :param n_points: amount of points you want to sample from distribution p
    :param n_dims: amount of dimensions of the points which are sampled
    :return: array with shape (n_points, n_dims)
    """

    samples = np.zeros((n_points, n_dims))
    x = np.zeros(n_dims)

    for i in range(n_points):
        if i % (n_points / 4) == 0:
            print(i)
        new_x = x + np.random.normal(size=n_dims, scale=0.1)

        counter = 0
        if accept_exponent(current=objective_function(x, data, labels, 0.01),
                           candidate=objective_function(new_x, data, labels, 0.01)):
            samples[i] = new_x
            x = new_x
        else:
            samples[i] = x

    return samples


def accept_exponent(current, candidate, factor=1):
    difference = candidate - current
    a = np.exp(- factor * difference)

    if a >= 1:
        return True
    elif np.random.rand() <= a:
        return True
    else:
        return False


def accept(p, q, sample_x, current_state):
    """
    Calculate the acceptance ratio between sample_x x' and current_state x
    according to the distributions p and q, and returns the proper parameter
    :param p: distribution to sample from
    :param q: distribution picked by the metropolis_hastings method
    :param sample_x: newly sampled x-coordinate to check if it suffices to the
    chosen distribution p
    :param current_state: old sample which has been previously accepted by
    the metropolis hastings algorithm.
    :return: True or False value if the sample_x is accepted
    """
    a = p(sample_x) * q(current_state, sample_x) / \
        (p(current_state) * q(sample_x, current_state))

    if a >= 1:
        return True
    elif np.random.rand() <= a:
        return True
    else:
        return False
