from week_3 import gen_sets
import plottebakkers.plot_IM_MacKayfig3111 as plotIM
import pickle

n = 50
method = "factor"
path_im, path_res = gen_sets.gen_paths(n, method)

with open(path_res, "rb") as f:
    temps = pickle.load(f)
    mes = pickle.load(f)
    stds = pickle.load(f)


plotIM.three_columns(temps, mes, stds)
