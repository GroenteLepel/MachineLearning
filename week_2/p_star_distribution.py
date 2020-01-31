import numpy as np


def p_star_distribution(data, labels, weights, alpha=0.01):
    return np.exp(- objective_function(weights, data, labels, alpha))


def objective_function(weights, data, labels, alpha):
    return error_function(weights, data, labels) + alpha * regularizer(weights)


def error_function(weights, data, labels):
    y = logistic(data, weights)
    decrease_y = y == 1
    increase_y = y == 0
    y[increase_y] += 1e-9
    y[decrease_y] -= 1e-9
    return -(np.dot(labels, np.log(y)) + np.dot((1 - labels), np.log(1 - y)))


def logistic(data, weights):
    """
    sigmoid function which returns a value between 0 and 1 for each element
    in data corresponding to the prediction that the weights make.
    :param data: group of data points.
    :param weights: weight vector predicting a seperation line.
    :return: sigmoid value in array of len(data)
    """
    return 1 / (1 + np.exp(- np.dot(weights, data.transpose())))


def regularizer(weights):
    return 0.5 * (weights ** 2).sum()
