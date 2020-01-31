import numpy as np
from week_2.p_star_distribution import objective_function


def normal(x, mean=0):
    """
    :return: float, probability that x is in a 3D-gaussian with identity matrix
    as covariance.
    """

    return np.exp(-1 * np.dot((x - mean), (x - mean)) / 2) / \
           (np.sqrt(2 * np.pi) ** 3)


def metropolis_hastings(p, data, labels, n_points=1000, n_dims=3, spread=0.1):
    """
    Method for sampling data from a distribution p.
    :param p: distribution to sample using this method.
    :param n_points: amount of points you want to sample from distribution p.
    :param n_dims: amount of dimensions of the points which are sampled.
    :param spread: determines the spread of the gaussian from which to draw
    samples.
    :return: array with shape (n_points, n_dims).
    """

    samples = np.zeros((n_points, n_dims))
    x = np.ones(n_dims)
    a_vals = np.zeros(n_points)

    print("Starting Metropolis Hastings sampling method for {} samples."
          .format(n_points))
    print("Spread sigma of proposal distribution:", spread)
    print("===============")
    print("|", end='')
    for i in range(n_points):
        if i / n_points * 100 % 5 == 0:
            print("█", end='')
        new_x = x + np.random.normal(size=n_dims, scale=spread)

        counter = 0
        a_vals[i] = accept_probability(
            current=objective_function(x, data, labels, 0.01),
            candidate=objective_function(new_x, data, labels, 0.01)
        )
        if accept_val(a_vals[i]):
            samples[i] = new_x
            x = new_x
        else:
            samples[i] = x

    print("|")

    return samples, a_vals


def accept_exponent(current, candidate, factor=1):
    difference = candidate - current
    a = np.exp(- factor * difference)

    if a >= 1:
        return True
    elif np.random.rand() <= a:
        return True
    else:
        return False


def accept_probability(current, candidate, factor=1):
    """
    Same as accept_exponent, but returns the accept value for analysis.
    """
    difference = candidate - current
    a = np.exp(- factor * difference)
    return a


def accept_val(a):
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
