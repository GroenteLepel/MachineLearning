import numpy as np
from scipy.stats import multivariate_normal


def normal(x, mean=0):
    """
    :return: float, probability that x is in a 3D-gaussian with identity matrix
    as covariance.
    """

    return np.exp(-1 * np.dot((x - mean), (x - mean)) / 2) / \
           (np.sqrt(2 * np.pi) ** 3)


def metropolis_hastings(p, n_points=1000, n_dims=3):
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
        new_x = np.random.normal(size=n_dims)

        counter = 0
        while not accept(p, normal, new_x, x):
            new_x = np.random.normal(size=n_dims)
            counter += 1
            if counter == 100000:
                print("overflowing at", i)

        samples[i] = new_x
        x = new_x

    return samples


def accept(p, q, sample_x, current_state):
    """
    Calculate the acceptance ratio between sample_x x' and current_state x
    according to the distributions p and q, and returns the proper parameter
    :param p: distribution to sample
    :param q: distribution picked by the metropolis_hastings method
    :param sample_x: newly sampled x-coordinate to check if it suffices to the
    chosen distribution p
    :param current_state: old sample which has been previously accepted by
    the metropolis hastings algorithm.
    :return: True or False value if the sample_x is accepted
    """
    p_star = p(sample_x)
    p_test = p(current_state)
    q_test = q(0, sample_x)
    q_star = q(sample_x, 0)

    a = p(sample_x) * q(current_state) / \
        (p(current_state) * q(sample_x))

    if a >= 1:
        return True
    elif np.random.rand() <= a:
        return True
    else:
        return False
