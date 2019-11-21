import numpy as np


def p_star_distribution(weights, data, labels):
    return np.exp(- objective_function(weights, data, labels))


def objective_function(weights, data, labels, alpha=0.01):
    return error_function(weights, data, labels) + alpha * regularizer(weights)


def error_function(weights, data, labels):
    y = logistic(data, weights)
    return np.dot(labels, np.log(y)) + np.dot((1 - labels), np.log(1 - y))


def logistic(data, weights):
    return 1 / (1 + np.exp(- np.dot(weights, data.transpose())))


def regularizer(weights):
    return 0.5 * (weights ** 2).sum()