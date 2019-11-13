import numpy as np
import matplotlib.pyplot as plt

# %% Description of the exercise
# Implement Gibbs sampling for the inference of a single one-dimensional
# Gaussian which we studied using maximuml ikelihood in section22.1. Assign a
# broad Gaussian prior toand a broad gamma prior (24.2) to the precision
# parameter= 1=2. Each update ofwill involve a sample from a Gaussian
# distribution,and each update ofrequires a sample from a gammad istribution.

# Sample from a 1D Gaussian
# The variance and mean need to be sampled. The variance is independent of the
# mean. The mean is dependent of your variance! With very small variance, the
# area of where your mean can be is much smaller.

# You sample the mean based on two conditionals:
#  for i in range(__):
#       x_i = p_1(__|__)
#       T_i = p_2(__|__)

# You use the mean and variance to

# Derive the conditionals from the first section on the wiki-page.

# %% Trying to work it out
# The mean is sampled from a gaussian. The mean for this gaussian is fixed.
fixed_mean = 0

# Initialize parameters for gamma function from which stdev is sampled.
shape_param = 7.5
scale_param = 1.0

# Derive a mean from a gamma distribution, and variance from a normal
#  distribution
sample_size = int(1e5)
mean_samples = np.zeros(sample_size)
beta_samples = np.zeros(sample_size)
for i in range(sample_size):
    beta_samples[i] = np.random.gamma(shape_param, scale_param)
    mean_samples[i] = np.random.normal(fixed_mean, 1/np.sqrt(beta_samples[i]))

plt.scatter(mean_samples, 1/np.sqrt(beta_samples), marker='.')
plt.title('Mean vs Standard deviation samples')
plt.xlabel('Mean')
plt.ylabel('Standard deviation')
plt.show()
