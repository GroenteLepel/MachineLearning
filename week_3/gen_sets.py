import pickle
import numpy as np
from week_3.ising_model import IsingModel
from week_3.ising_optimiser import IsingOptimiser
import plottebakkers.plot_IM_MacKayfig3111 as plotIM


def gen_markov_set(io: IsingOptimiser, length_chain: int):
    cnt, betas, me, std = \
        io.simulated_annealing(beta_init=1/500,
                               length_markov_chain=length_chain)
    temp = 1 / betas
    me /= 100
    std /= 100

    return temp, std, me


def gen_initbeta_set(io: IsingOptimiser, beta: float):
    cnt, betas, me, std = io.simulated_annealing(beta_init=beta,
                                                 cooling_factor=1.09)
    temp = 1 / betas
    me /= 100
    std /= 100
    return temp, std, me


def gen_factor_set(io: IsingOptimiser, factor: float):
    cnt, betas, me, std = io.simulated_annealing(cooling_factor=factor,
                                                 beta_init=1/500)
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
    beta_i = np.array([1 / 200, 1 / 300, 1/400])
    temps, stds, mes = [], [], []
    for b in beta_i:
        t, std, me = gen_initbeta_set(io, b)
        temps.append(t)
        stds.append(std)
        mes.append(me)
        io.reset()
    return temps, stds, mes


def gen_full_factor_set(io: IsingOptimiser):
    factors = np.array([1.05, 1.08, 1.1])
    temps, stds, mes = [], [], []
    for f in factors:
        t, std, me = gen_factor_set(io, f)
        temps.append(t)
        stds.append(std)
        mes.append(me)
        io.reset()
    return temps, stds, mes


def gen_paths(n: int, method: str):
    main_path = "../data/picklejar/"
    fin = "im_frustrated_n{}.pkl".format(n)
    fout = "sa_{}_n{}.pkl".format(method, n)
    path_in = "{}{}".format(main_path, fin)
    path_out = "{}{}".format(main_path, fout)
    return path_in, path_out


def gen_sets(method: str, n_spins: int = 50,
             pickle_result: bool = True, plot_result: bool = False):
    pathin, pathout = gen_paths(n_spins, method)

    with open(pathin, 'rb') as f:
        im_fr = pickle.load(f)

    im_fr.remake_coupling_matrix(std=5)
    io_fr = IsingOptimiser(im_fr, neighbourhood=2)

    if method == "markov":
        temps, stds, mes = gen_full_markov_set(io_fr)
    elif method == "beta":
        temps, stds, mes = gen_full_beta_set(io_fr)
    elif method == "factor":
        temps, stds, mes = gen_full_factor_set(io_fr)

    if pickle_result:
        with open(pathout, 'wb') as output:
            pickle.dump(temps, output)
            pickle.dump(mes, output)
            pickle.dump(stds, output)

    if plot_result:
        plotIM.three_columns(temps, mes, stds)
