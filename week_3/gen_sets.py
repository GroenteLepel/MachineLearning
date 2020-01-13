import pickle
import numpy as np
import week_3.params as params
from week_3.ising_model import IsingModel
from week_3.ising_optimiser import IsingOptimiser
import plottebakkers.plot_IM_MacKayfig3111 as plotIM


def gen_markov_set(io: IsingOptimiser, length_chain: int):
    cnt, betas, me, std = \
        io.simulated_annealing(length_markov_chain=length_chain)
    temp = 1 / betas
    me /= 100
    std /= 100
    im = io.im

    return temp, std, me, im


def gen_initbeta_set(io: IsingOptimiser, beta: float):
    cnt, betas, me, std = io.simulated_annealing(beta_init=beta,
                                                 cooling_factor=1.09)
    temp = 1 / betas
    me /= 100
    std /= 100
    im = io.im
    return temp, std, me, im


def gen_factor_set(io: IsingOptimiser, factor: float):
    cnt, betas, me, std = io.simulated_annealing(cooling_factor=factor)
    temp = 1 / betas
    me /= 100
    std /= 100
    im = io.im
    return temp, std, me, im


def gen_full_markov_set(io: IsingOptimiser):
    lengths = params.MARKOVS
    temps, stds, mes, ims = [], [], [], []
    for l in lengths:
        t, std, me, im = gen_markov_set(io, l)
        temps.append(t)
        stds.append(std)
        mes.append(me)
        ims.append(im)
        io.reset()
    return temps, stds, mes, ims


def gen_full_beta_set(io: IsingOptimiser):
    beta_i = params.BETAS
    temps, stds, mes, ims = [], [], [], []
    for b in beta_i:
        t, std, me, im = gen_initbeta_set(io, b)
        temps.append(t)
        stds.append(std)
        mes.append(me)
        ims.append(im)
        io.reset()
    return temps, stds, mes, ims


def gen_full_factor_set(io: IsingOptimiser):
    factors = params.FACTORS
    temps, stds, mes, ims = [], [], [], []
    for f in factors:
        t, std, me, im = gen_factor_set(io, f)
        temps.append(t)
        stds.append(std)
        mes.append(me)
        ims.append(im)
        io.reset()
    return temps, stds, mes, ims


def gen_paths(n: int, method: str):
    main_path = "../data/picklejar/"
    fin = "im_frustrated_n{}.pkl".format(n)
    fout = "sa_{}_n{}.pkl".format(method, n)
    path_in = "{}{}".format(main_path, fin)
    path_out = "{}{}".format(main_path, fout)
    return path_in, path_out


def gen_io(method: str, pathin: str):

    with open(pathin, 'rb') as f:
        im_fr = pickle.load(f)

    io = IsingOptimiser(im_fr, neighbourhood=2)
    return io


def gen_sets(method: str,
             pickle_result: bool = True, plot_result: bool = False):
    pathin, pathout = gen_paths(params.N_SPINS, method)
    io_fr = gen_io(method, pathin)

    if method == "markov":
        temps, stds, mes, ims = gen_full_markov_set(io_fr)
    elif method == "beta":
        temps, stds, mes, ims = gen_full_beta_set(io_fr)
    elif method == "factor":
        temps, stds, mes, ims = gen_full_factor_set(io_fr)
    else:
        raise ValueError("No proper method given.")

    if pickle_result:
        with open(pathout, 'wb') as output:
            pickle.dump(temps, output)
            pickle.dump(mes, output)
            pickle.dump(stds, output)
            pickle.dump(ims, output)

    if plot_result:
        plotIM.three_columns(temps, mes, stds, method)

    return temps, mes, stds, ims


def find_min_spread(method: str, steps: int = 10):
    minima = np.zeros((steps, 3))
    pathin, pathout = gen_paths(params.N_SPINS, method)
    io_fr = gen_io(method, pathin)

    for i in range(steps):
        if method == "markov":
            temps, stds, mes, ims = gen_full_markov_set(io_fr)
            for j in range(3):
                minima[i][j] = mes[j][-1]
        elif method == "beta":
            temps, stds, mes, ims = gen_full_beta_set(io_fr)
            for j in range(3):
                minima[i][j] = mes[j][-1]
        elif method == "factor":
            temps, stds, mes, ims = gen_full_factor_set(io_fr)
            for j in range(3):
                minima[i][j] = mes[j][-1]
        else:
            raise ValueError("No proper method given.")

    return minima
