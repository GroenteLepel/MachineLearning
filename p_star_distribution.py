import numpy as np


def p_star_distribution(data, labels, weights, alpha=0.01):
    # TODO: Calculate only the powers in the exponents, not the exp itself?
    return np.exp(- objective_function(weights, data, labels, alpha))


def objective_function(weights, data, labels, alpha):
    return error_function(weights, data, labels) + alpha * regularizer(weights)


def error_function(weights, data, labels):
    y = logistic(data, weights)
    # TODO: adjust the calculation so we work with what is in the power of the
    #  exponents, and remove this ugly adjustment of y.
    decrease_y = y == 1
    increase_y = y == 0
    y[increase_y] += 1e-9
    y[decrease_y] -= 1e-9
    return -(np.dot(labels, np.log(y)) + np.dot((1 - labels), np.log(1 - y)))


def logistic(data, weights):
    # TODO: Calculate only the powers in the exponents, not the exp itself?
    return 1 / (1 + np.exp(- np.dot(weights, data.transpose())))


def regularizer(weights):
    return 0.5 * (weights ** 2).sum()
