from week_3.ising_optimiser import IsingOptimiser
import plottebakkers.plot_IM_MacKayfig3111 as plotIM
import numpy as np
import matplotlib.pyplot as plt
import pickle


# N = 500
# neighbourhoods = np.arange(1, 6)
#
# im_fr = IsingModel(n=N, frustrated=True, threshold=False)


def gen_markov_set(io: IsingOptimiser, length_chain: int):
    beta_i = 1 / 100
    cnt, betas, me, std = \
        io.simulated_annealing(beta_init=beta_i,
                               length_markov_chain=length_chain)
    temp = 1 / betas
    me /= 100
    std /= 100

    return temp, std, me


def gen_initbeta_set(io: IsingOptimiser, beta: float):
    cnt, betas, me, std = io.simulated_annealing(beta_init=beta)
    temp = 1 / betas
    me /= 100
    std /= 100
    return temp, std, me


def gen_factor_set(io: IsingOptimiser, factor: float):
    cnt, betas, me, std = io.simulated_annealing(cooling_factor=factor)
    temp = 1 / betas
    me /= 100
    std /= 100
    return temp, std, me


def gen_full_markov_set(io: IsingOptimiser):
    lengths = np.array([1000, 2000, 3000])
    temps, stds, mes = [], [], []
    for l in lengths:
        t, std, me = gen_markov_set(io, l)
        temps.append(t)
        stds.append(std)
        mes.append(me)
        io.reset()
    return temps, stds, mes


def gen_full_beta_set(io: IsingOptimiser):
    beta_i = np.array([1 / 100, 1 / 10, -1])
    temps, stds, mes = [], [], []
    for b in beta_i:
        t, std, me = gen_initbeta_set(io, b)
        temps.append(t)
        stds.append(std)
        mes.append(me)
        io.reset()
    return temps, stds, mes


def gen_full_factor_set(io: IsingOptimiser):
    factors = np.array([1.01, 1.5, 2])
    temps, stds, mes = [], [], []
    for f in factors:
        t, std, me = gen_factor_set(io, f)
        temps.append(t)
        stds.append(std)
        mes.append(me)
        io.reset()
    return temps, stds, mes


with open('../data/picklejar/im_frustrated_n50.pkl', 'rb') as f:
    im_fr = pickle.load(f)

io_fr = IsingOptimiser(im_fr, neighbourhood=2)

# [t_1, t_2, t_3], [std_1, std_2, std_3], [me_1, me_2, me_3] = \
#     gen_full_markov_set(io_fr)

# temps, stds, mes = gen_full_beta_set(io_fr)
temps, stds, mes = gen_full_factor_set(io_fr)

# with open('../data/picklejar/sa_markov_run_n50.pkl', 'wb') as output:
#     pickle.dump(t_1, output)
#     pickle.dump(std_1, output)
#     pickle.dump(me_1, output)
#
#     pickle.dump(t_2, output)
#     pickle.dump(std_2, output)
#     pickle.dump(me_2, output)
#
#     pickle.dump(t_3, output)
#     pickle.dump(std_3, output)
#     pickle.dump(me_3, output)

plotIM.three_columns(temps, mes, stds)
