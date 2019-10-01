import numpy as np

# %% Defining functions
def sigmoid(x):
    return (1 + np.e ** (-x)) ** (-1)


def p(x, w):
    return sigmoid(x.dot(w))


def E(w, X, T):
    Y = p(X, w)
    N = len(T)
    return float(-1. / N * (
            np.transpose(T).dot((np.log(Y))) + np.transpose(1 - T).dot((np.log(1 - Y)))))


def Edecay(w, X, T):
    Y = p(X, w)
    N = len(T)
    return float(-1. / N * (
            np.transpose(T).dot((np.log(Y))) + np.transpose(1 - T).dot((np.log(1 - Y))))) + np.sum(w ** 2) * 1. / len(w)


def grad(w, X, T):  # the derivative of E2 with respect to w_{i}
    N = len(T)
    diff = (p(X, w) - T)
    return np.transpose(1. / N * np.transpose(diff).dot(X))


def graddecay(w, X, T, lab):
    N = len(T)
    diff = (p(X, w) - T)
    return np.transpose(1. / N * np.transpose(diff).dot(X)) + lab * w / len(w)


def Hessian(w, X, T):
    Y = p(X, w)
    N = len(T)
    return 1. / N * (np.transpose(X).dot((((1 - Y) * (Y) * X))))


def Hessiandecay(w, X, T, lab):
    Y = p(X, w)
    N = len(T)
    return 1. / N * (np.transpose(X).dot((((1 - Y) * (Y) * X)))) + np.identity(N) * lab * 1. / len(w)


def entropy(Xtrain, Ttrain, Xtest, Ttest, eta, epochs=10000):
    # initializing constants
    w = np.random.uniform(-0.01, 0.01, (784, 1))
    Trainloss = np.zeros(epochs)
    Testloss = np.zeros(epochs)

    for l in range(0, epochs):
        if l % int(epochs / 4) == 0:
            print('{0:d}% done with eta={1:4.2f}'.format(int(l / (epochs / 4)),
                                                         eta))
        dw = -eta * grad(w, Xtrain, Ttrain)
        w = w + dw
        Trainloss[l] = E(w, Xtrain, Ttrain)
        Testloss[l] = E(w, Xtest, Ttest)

    return Trainloss, Testloss


def decay(Xtrain, Ttrain, Xtest, Ttest, eta, epochs=10000):
    # This function is not correct yet, just a copy of entropy
    # initializing constants
    w = np.random.uniform(-0.01, 0.01, (784, 1))
    Trainloss = np.zeros(epochs)
    Testloss = np.zeros(epochs)

    for l in range(0, epochs):
        if l % int(epochs / 4) == 0:
            print('{0:d}% done with eta={1:4.2f}'.format(int(l / (epochs / 4)),
                                                         eta))
        dw = -eta * grad(w, Xtrain, Ttrain)
        w = w + dw
        Trainloss[l] = E(w, Xtrain, Ttrain)
        Testloss[l] = E(w, Xtest, Ttest)

    return Trainloss, Testloss


def momentum(Xtrain, Ttrain, Xtest, Ttest, eta, alpha, epochs):
    # initializing constants
    dw = 0
    w = np.random.uniform(-0.01, 0.01, (784, 1))
    Trainloss = np.zeros(epochs)
    Testloss = np.zeros(epochs)

    for l in range(0, epochs):
        if l % int(epochs / 4) == 0:
            print('{0:d}% done with eta={1:4.2f}'.format(int(l / (epochs / 4)),
                                                         eta))

        dw = -eta * grad(w, Xtrain, Ttrain) + alpha * dw
        w = w + dw
        Trainloss[l] = E(w, Xtrain, Ttrain)
        Testloss[l] = E(w, Xtest, Ttest)

    return Trainloss, Testloss


def weightdecay(Xtrain, Ttrain, Xtest, Ttest, eta, alpha, lab, epochs):
    # initializing constants
    dw = 0
    w = np.random.uniform(-0.01, 0.01, (784, 1))
    Trainloss = np.zeros(epochs)
    Testloss = np.zeros(epochs)

    for l in range(0, epochs):
        if l % int(epochs / 4) == 0:
            print('{0:d}% done with eta={1:4.2f}'.format(int(l / (epochs / 4)),
                                                         eta))
        dw = -eta * graddecay(w, Xtrain, Ttrain, lab) + alpha * dw
        w = w + dw
        Trainloss[l] = E(w, Xtrain, Ttrain)
        Testloss[l] = E(w, Xtest, Ttest)

    return Trainloss, Testloss