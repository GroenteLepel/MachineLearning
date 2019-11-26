import numpy as np


def normal(x, mean):
    """
    :param x:
    :param mean:
    :param stdev:
    :return: float, probability that x is in a gaussian.
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
        new_x = x + np.random.normal(size=n_dims)

        counter = 0
        while not accept(p, normal, new_x, x):
            new_x = x + np.random.normal(size=n_dims)
            counter += 1
            if counter == 1000:
                print('wtf')
                print(new_x)
                print(x)

        samples[i] = new_x
        x = new_x

    return samples


def accept(p, q, sample_x, current_state):
    """
    Calculate the accaptence ratio between sample_x x' and current_state x
    according to the distributions p and q, and returns the proper parameter
    :param p: distribution to sample
    :param q: distribution picked by the metropolis_hastings method
    :param sample_x:
    :param current_state:
    :return:
    """

    a = p(sample_x) * q(current_state, sample_x) / \
        (p(current_state) * q(sample_x, current_state))

    if a >= 1:
        return True
    elif np.random.rand() <= a:
        return True
    else:
        return False
