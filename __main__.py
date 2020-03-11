import sys
import os
current_path = os.getcwd()
sys.path.extend([current_path])

# for mcmc
# exec(open('week_2/run_mcmc.py').read())

# for simulated annealing
# exec(open('week_3/run_ising.py').read())

# for boltzmann
# exec(open('week_6/run_boltzmann.py').read())

# for control theory
# exec(open('week_9/run_arm.py').read())