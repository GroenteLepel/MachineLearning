import sys
import os
current_path = os.getcwd()
sys.path.extend([current_path])

# for mcmc
exec(open('week_2/run_mcmc.py').read())
