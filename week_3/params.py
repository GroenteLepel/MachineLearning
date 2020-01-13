import numpy as np
global MARKOVS
global FACTORS
global BETAS

N_SPINS = 50
MARKOVS = np.array([100, 1000, 5000])
BETAS = np.array([1 / 20, 1 / 100, 1/500])
FACTORS = np.array([1.01, 1.09, 1.5])