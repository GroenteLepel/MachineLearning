#%% Imports
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

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


#%% declaring functions
def variance_weights(stdev_Q):
    return stdev_Q**2/(2*stdev_Q**2-1)**(1/2) - 1


#%% plot a
N = 60
Q_stdevs = np.linspace(0.1,1.6,N)
Q_stdevs_theory = np.linspace(1/np.sqrt(2)+1e-6,1.6,N)
R_3, R_4, R_5 = np.zeros(N), np.zeros(N), np.zeros(N)
D_3, D_4, D_5 = np.zeros(N), np.zeros(N), np.zeros(N)
weights30 = np.zeros((N,30))
for i, Q_stdev in enumerate(Q_stdevs):
    np.random.seed(1234567890)
    rnd_numbers = np.random.normal(0,Q_stdev,size=int(1e5))
    weights = scipy.stats.norm(0,1).pdf(rnd_numbers)/scipy.stats.norm(0,Q_stdev).pdf(rnd_numbers)
    R_3[i] = np.average(weights[:1000])
    R_4[i] = np.average(weights[:10000])
    R_5[i] = np.average(weights[:100000])
    D_3[i] = np.std(weights[:1000])
    D_4[i] = np.std(weights[:10000])
    D_5[i] = np.std(weights[:100000])
    weights30[i] = weights[:30]
    
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.plot(Q_stdevs, R_3, label='1.000')
ax1.plot(Q_stdevs, R_4, label='10.000')
ax1.plot(Q_stdevs, R_5, label='100.000')
ax1.plot(Q_stdevs, np.linspace(1,1,N), label='theory')
ax1.set_title('Average weights vs $\sigma_Q$')
ax1.set_xlabel('$\sigma_Q$')
ax1.set_ylabel('Average weights')
ax1.legend()

#%% plot b
ax2.plot(Q_stdevs, D_3, label='1.000')
ax2.plot(Q_stdevs, D_4, label='10.000')
ax2.plot(Q_stdevs, D_5, label='100.000')
ax2.plot(Q_stdevs_theory, np.sqrt(variance_weights(Q_stdevs_theory)), label='theory')
ax2.set_title('Standard deviation of weights vs $\sigma_Q$')
ax2.set_xlabel('$\sigma_Q$')
ax2.set_ylabel('Empirical standard deviation of weights')
ax2.set_ylim(0,10)
ax2.legend()

#%% plot c
for i in range(30):
    ax3.plot(Q_stdevs, weights30[:,i])
ax3.set_title('30 weights vs $\sigma_Q$')
ax3.set_xlabel('$\sigma_Q$')
ax3.set_ylabel('Weight')

plt.show()