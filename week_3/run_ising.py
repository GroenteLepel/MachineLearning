from week_3.ising_optimiser import IsingOptimiser
import plottebakkers.plot_IM_MacKayfig3111 as plotIM
import numpy as np
import matplotlib.pyplot as plt
import pickle


# N = 500
# neighbourhoods = np.arange(1, 6)
#
# im_fr = IsingModel(n=N, frustrated=True, threshold=False)


def genset(length_chain):
    cnt, betas, me, std = io_fr._simulated_annealing(
        length_markov_chain=length_chain)
    temp = 1 / betas
    me /= 100
    std /= 100

    return temp, std, me


with open('../data/im_frustrated_n500.pkl', 'rb') as f:
    im_fr = pickle.load(f)

io_fr = IsingOptimiser(im_fr, neighbourhood=2)

t_1, std_1, me_1 = genset(1000)
io_fr.reset()
t_2, std_2, me_2 = genset(2000)
io_fr.reset()
t_3, std_3, me_3 = genset(3000)
io_fr.reset()

with open('../data/sa_markov_run_n500.pkl', 'wb') as output:
    pickle.dump(t_1, output)
    pickle.dump(std_1, output)
    pickle.dump(me_1, output)

    pickle.dump(t_2, output)
    pickle.dump(std_2, output)
    pickle.dump(me_2, output)

    pickle.dump(t_3, output)
    pickle.dump(std_3, output)
    pickle.dump(me_3, output)

plotIM.three_columns(t_1, std_1, me_1,
                     t_2, std_2, me_2,
                     t_3, std_3, me_3)
