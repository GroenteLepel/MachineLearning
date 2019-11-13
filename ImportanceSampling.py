#%% Imports
import numpy as np

#%% Exercise description
# Have a gaussian for P(x) and Q(x) both with means 0 and stdevs std_p, std_q.
#  Sample x_q from Q(x), calculate the probability P(x_q), and use this to:
#
#  - calculate the weight of x_q: w_q = P(x_q) / Q(x_q)
#  - calculate the normalization constant Z = \int dx P(x)

# Afterwards, create a plot of:
#  - estimated normalization constant Z
#  - empirical standard deviation of w_R (?)
#  - 30 weights (?)

# The plots seem to run on the x-axis from 1 to 1.6, and contain runs with
#  R = 1.000, 10.000, 100.000.

#%% Function declerations
def prob_in_gaussian(x):
    probability = 
    return probability

def normal_pdf(x, mean, stdev):
    var = float(stdev)**2
    denom = (2 * np.pi * var) ** (1./2)
    num = np.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num/denom

#%% Init
mean_p, mean_q = 0, 0
stdev_p, stdev_q = 1, 2
n_points = int(1e3)


#%% 
x_q = np.random.normal(loc=mean_q, scale=stdev_q, size=n_points)
