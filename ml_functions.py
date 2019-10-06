# %% Defining functions
import numpy as np
import time
from scipy.optimize import minimize

# %% Global constants
RESOLUTION = 28  # resolution of an image
RES_SQ = RESOLUTION * RESOLUTION
MAX_INT = 255  # max intensity of a pixel


# %% functions
def sigmoid(x):
    return (1 + np.exp(-x)) ** (-1)


def probability(x, w):
    """
    probability function p(t = 0/1 | x) which uses a sigmoid function
    :param x: given event of which the probability needs to comply
    :param w: weight belonging to x
    :return:
    """
    return sigmoid(x.dot(w))


def loss_function(weights, coords, labels, decay_factor):
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
    n_points = len(labels)  # amount of data points
    if decay_factor != 0.0:
        decay_term = decay_factor * np.sum(weights ** 2) / (2 * len(weights))
    else:
        decay_term = 0.0
    return float(
        -1. / n_points * (
                np.transpose(labels).dot((np.log(y + 1e-9)))
                + np.transpose(1 - labels).dot((np.log(1 - y)))
        ) + decay_term
    )


def linesearch_loss_function(gamma, d, weights, coords, labels, decay_factor):
    """
    For the line search method we need to minimize E(w+gamma*d) where d=-gradE.
    To do this we wrote a function to return the above expression E(w+gamma*d).
    We can minimize this function using scipy.optimize.minimize module
    :return: E(w+gamma*d)
    """
    return loss_function(weights + gamma * d, coords, labels, decay_factor)


def gradient_function(weights, coords, labels, decay_factor):
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
    n_points = len(labels)  # amount of data points
    diff = (probability(coords, weights) - labels)
    return np.transpose(1. / n_points * np.transpose(diff).dot(coords)) \
           + decay_factor * weights / len(weights)


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
    :return: hessian
    """
    # shape (dxd, n), or (784, n)
    t_coords = np.transpose(coords)
    n_points = len(t_coords)

    # shape (n, 1)
    y = probability(coords, weights)
    if decay_factor != 0.0:
        # shape (dxd)
        n = np.arange(1, len(weights) + 1)
        decay_term = np.zeros((RES_SQ + 1, RES_SQ + 1))
        np.fill_diagonal(decay_term, decay_factor / n)
    else:
        decay_term = 0.0

    return 1. / n_points * (
        np.transpose(coords).dot(((1 - y) * y * coords))) + decay_term


def classification_check(coords, labels, weights):
    """
    :param weights:
    :param coords: pixels intensities
    :param labels: denotes 0 or 1 for classification of the data point
    :return: percentages of wrongly classified data (for training and test)
    """
    probabilities = probability(coords, weights)
    prob_bools = np.round(probabilities)
    equal_counter = (prob_bools == labels).sum()
    return 1 - (equal_counter / len(labels))


def gradient_descent(train_coords, train_labels, test_coords, test_labels,
                     step_strength=0.1, momentum_step=0.0, decay_factor=0.0,
                     newtonian=False, linesearch=False, congrad_descent=False,
                     epochs=10000, batch_size=-1):
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

    # make sure we have a decay factor if one wants to use netwon method
    while newtonian and decay_factor == 0.0:
        print('You want to use the Netwon method')
        decay_factor = float(
            input('Please give a non-zero decay-factor: ', end=''))

    # initializing constants
    dw = np.zeros((RES_SQ + 1, 1))
    train_loss, test_loss = np.zeros(epochs), np.zeros(epochs)
    indices = np.linspace(0, len(train_labels) - 1, len(train_labels),
                          dtype=int)

    w = np.random.normal(0, 1. / np.sqrt(RES_SQ + 1), (RES_SQ + 1, 1))

    print('Starting gradient descent using eta={0:4.2f}, alpha={1:4.2f}, '
          'decay_factor={2:4.2f}.'
          .format(step_strength, momentum_step, decay_factor))

    time_start = time.time()
    for l in range(0, epochs):
        if 4 * l % epochs == 0:
            print('{0:d}% done.'.format(int(100 * l / epochs)))

        if batch_size != -1:
            # If a batch size is given, create a batch of images from the total
            # n number of images to reduce training time.
            batch = np.sort(
                np.random.choice(indices, batch_size, replace=False))
            train_coords_sel = train_coords[batch]
            train_labels_sel = train_labels[batch]
        else:
            train_coords_sel = train_coords
            train_labels_sel = train_labels

        gradE_prev = None
        if newtonian:
            hessian_matrix = hessian(w, train_coords, decay_factor)

            hessian_inv = np.linalg.inv(hessian_matrix)

            dw = -hessian_inv.dot(gradient_function(w, train_coords_sel,
                                                    train_labels_sel,
                                                    decay_factor))
        elif linesearch or congrad_descent:
            gradE = gradient_function(w, train_coords_sel, train_labels_sel,
                                      decay_factor)
            if gradE_prev != None:
                beta = np.dot((gradE - gradE_prev)[:, 0], gradE[:, 0]) / (
                    np.linalg.norm(gradE_prev)) ** 2
                d = -gradE + beta * d_prev

            else:
                d = -gradE
            gamma_opt = float(minimize(linesearch_loss_function, 1e-4,
                                       args=(
                                       d, w, train_coords_sel, train_labels_sel,
                                       decay_factor)).x)
            dw = gamma_opt * d
            if congrad_descent:
                gradE_prev = gradE
                d_prev = d
        else:
            dw = -step_strength \
                 * gradient_function(w, train_coords_sel, train_labels_sel,
                                     decay_factor) \
                 + momentum_step * dw

        w = w + dw

        train_loss[l] = loss_function(w, train_coords_sel, train_labels_sel,
                                      decay_factor)

        test_loss[l] = loss_function(w, test_coords, test_labels,
                                     decay_factor)

    print('done in {0:3.2f} seconds'.format(time.time() - time_start))

    return train_loss, test_loss, w
