import numpy as np
import week_2.p_star_distribution as p_star


def metropolis_hastings(data, labels, n_points=1000, n_dims=3, spread=0.1):
    """
    Method for sampling data from a distribution p.
    :param data: data points used to train the data on.
    :param labels: labels corresponding to the data points above.
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

        a_vals[i] = accept_probability(
            current=p_star.objective_function(x, data, labels, 0.01),
            candidate=p_star.objective_function(new_x, data, labels, 0.01)
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
