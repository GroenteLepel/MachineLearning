# %% Defining functions
import numpy as np
import time

# %% Global constants
RESOLUTION = 28  # resolution of an image
MAX_INT = 255  # max intensity of a pixel

# %% functions
def sigmoid(x):
    return (1 + np.e ** (-x)) ** (-1)


def probability(x, w):
    """
    probability function p(t = 0/1 | x) which uses a sigmoid function
    :param x: given event of which the probability needs to comply
    :param w: weight belonging to x
    :return:
    """
    return sigmoid(x.dot(w))


def loss_function(weights, coords, label, decay_factor):
    """
    Loss function which is the goal to be minimized.
    :param weights: weights connected to each x value
    :param coords: N-dimensional coordinates for each data point
    :param label: label given to each data point with coordinates "coords", is
    either 0 or 1.
    :param decay_factor: factor indicating how strong the decay must be
    implemented in the determination of finding the minimum.
    :return:
    """
    y = probability(coords, weights)
    n_points = len(label)  # amount of data points
    if decay_factor != 0.0:
        decay_term = decay_factor * np.sum(weights ** 2) / (2 * len(weights))
    else:
        decay_term = 0.0
    return float(
        -1. / n_points * (
                np.transpose(label).dot((np.log(y)))
                + np.transpose(1 - label).dot((np.log(1 - y)))
        ) + decay_term
    )


def gradient_function(weights, coords, label, decay_factor):
    """
    Gradient (derivative) of the loss_function with respect to the weight for
    each data point (w_i). This is used for determining the direction towards
    the a minimum in E(w) (loss_function).
    :param weights: weight belonging to data point with coordinates X and its
    label
    :param coords: N-dimensional coordinates for each data point
    :param label: label given to each data point with coordinates "coords", is
    either 0 or 1.
    :param decay_factor: factor indicating the strength of the decay method to
    determine the minimum in loss. Set to zero if you don't want to include this
    :return:
    """
    n_points = len(label)  # amount of data points
    diff = (probability(coords, weights) - label)
    return np.transpose(1. / n_points * np.transpose(diff).dot(coords)) \
           + decay_factor * weights / len(weights)


# TODO: implement
def hessian(weights, coords, decay_factor):
    """
    Hessian information used to optimize finding a minimum in E
    (loss_function). By making a quadratic approximation to E around w, the
    Hessian is connected to the second order approximation.
    :param weights: weight belonging to data point with coordinates X and its
    label
    :param coords: N-dimensional coordinates for each data point
    :param decay_factor: factor indicating the strength of the decay method to
    determine the minimum in loss. Set to zero if you don't want to include this
    :return:
    """
    # shape (dxd, n), or (784, n)
    t_coords = np.transpose(coords)
    n_points = len(t_coords)

    # shape (n, 1)
    y = probability(coords, weights)

    if decay_factor != 0.0:
        # shape (dxd)
        decay_term = np.identity(RESOLUTION*RESOLUTION) \
                     * decay_factor / len(weights)
    else:
        decay_term = 0.0

    # if you want to calc hessian for element i, j, so H_ij:
    i, j = 0, 1  # just for psuedocode-sake

    # this is element i, j of hessian matrix:
    H = 1. / n_points * (t_coords[i] * y * (1 - y) * t_coords[j]).sum()

    return H
    # return 1. / n_points * (
    #     np.transpose(coords).dot(((1 - y) * y * coords))) + decay_term


def classification_check(coords, labels, weights):
    """
    :coords: pixels intensities
    :labels: denotes 0 or 1 for classification of the datapoint
    :return: percentages of wrongly classified data (for training and test)
    """
    probabilities = probability(coords, weights)
    prob_bools = np.round(probabilities)
    equal_counter = (prob_bools == labels).sum()
    return 1 - (equal_counter / len(labels))


def gradient_descent(train_coords, train_labels, test_coords, test_labels,
                     step_strength=0.0, momentum_step=0.0, decay_factor=0.0,
                     newtonian=False, epochs=10000, batch_size=-1):
    """
    Function performing the method of gradient descent by initializing a random
    w and updating it according to the gradient (gradient_function) of the
    entropy (loss_function).
    :param train_coords: coordinates belonging to the data on which the network
    needs to be trained.
    :param train_labels: labels belonging to the data on which the network
    needs to be trained.
    :param test_coords: coordinates belonging to the data on which the network
    needs to be tested.
    :param test_labels: labels belonging to the data on which the network
    needs to be tested.
    :param step_strength: step size determining how big of a step has to be
    taken in the direction of the minimum (eta).
    :param momentum_step: default 0. If wanted to include the momentum, this
    uses the previously calculated dw step to give a momentum to the learning
    process, potentially optimizing it.
    :param newtonian: boolean indicating if the gradient descent should be
    determined via the newtonian method.
    :param epochs: amount of times the w has to be adjusted before ending the
    learning
    :param decay_factor: factor indicating the strength of the decay method to
    determine the minimum in loss. Set to zero if you don't want to include
    this.
    :param batch_size: every epoch a random subset of size batch_size of the 
    data is used to calculate the gradient. 
    :return: The loss in the training and testing, adn the hyperplane generated
    from training. The smaller the losses are, the better the training went.
    Should converge relative to the number of epochs.
    """
    # initializing constants
    if batch_size == -1:
        batch_size = len(train_labels)
    dw = np.zeros((784, 1))
    w = np.random.normal(0, 1. / np.sqrt(784), (784, 1))
    train_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)
    print('Starting gradient descent using eta={0:4.2f}, alpha={1:4.2f}, '
          'decay_factor={2:4.2f}.'
          .format(step_strength, momentum_step, decay_factor))
    indices = np.linspace(0, len(train_labels) - 1, len(train_labels),
                          dtype=int)
    time_start = time.time()
    for l in range(0, epochs):
        selectedindices = np.sort(
            np.random.choice(indices, batch_size, replace=False))
        train_coordsbatch = train_coords[selectedindices]
        train_labelsbatch = train_labels[selectedindices]
        if l % int(epochs / 4) == 0:
            print('{0:d}% done.'.format(
                int(l / (epochs / 4) * 25)))
        if newtonian:
            # TODO: implement correctly. The assignment says to use the inverse
            #  of the Hessian, but I do not know how to do this in the array
            #  form that is used.
            dw = - np.transpose(hessian(w,
                                        train_coords,
                                        decay_factor)
                                ) \
                .dot(gradient_function(w, train_coords, train_labels,
                                       decay_factor))
        else:
            dw = -step_strength \
                 * gradient_function(w, train_coords, train_labels,
                                     decay_factor) \
                 + momentum_step * dw

        dw = -step_strength \
             * gradient_function(w, train_coordsbatch, train_labelsbatch,
                                 decay_factor) \
             + momentum_step * dw
        w = w + dw
        train_loss[l] = loss_function(w, train_coords, train_labels,
                                      decay_factor)
        test_loss[l] = loss_function(w, test_coords, test_labels,
                                     decay_factor)

    print('done in {0:3.2f} seconds'.format(time.time() - time_start))

    return train_loss, test_loss, w
