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