from week_3 import gen_sets
from week_3.ising_model import IsingModel
from week_3.ising_optimiser import IsingOptimiser
import plottebakkers.plot_IM_MacKayfig3111 as plotIM
import pickle

#%% Recreating McKay
# picklejar = "../data/picklejar/"
#
# file = "im_ferromagnetic_n50.pkl"
# print("{}{}".format(picklejar, file))
# with open("{}{}".format(picklejar, file), 'rb') as input:
#     im_fer = pickle.load(input)
#
# file = "im_frustrated_n50.pkl"
# with open("{}{}".format(picklejar, file), 'rb') as input:
#     im_fr = pickle.load(input)
#
# io_fer = IsingOptimiser(im_fer, 2)
# io_fr = IsingOptimiser(im_fr, 2)
#
# cnt, beta, me, stde = io_fer.optimise('sa')
# file = "io_fer_solved.pkl"
# with open("{}{}".format(picklejar, file), 'wb') as output:
#     pickle.dump(beta, output)
#     pickle.dump(me, output)
#     pickle.dump(stde, output)
#
# cnt, beta, me, stde = io_fr.optimise('sa')
# file = "io_fr_solved.pkl"
# with open("{}{}".format(picklejar, file), 'wb') as output:
#     pickle.dump(beta, output)
#     pickle.dump(me, output)
#     pickle.dump(stde, output)

plotIM.plot_paramtest("markov")
plotIM.plot_paramtest("beta")
plotIM.plot_paramtest("factor")

#%% Testing parameters
# temps, mes, stds, ims = gen_sets.gen_sets("markov", plot_result=True)
# temps, mes, stds, ims = gen_sets.gen_sets("factor", plot_result=True)
# temps, mes, stds, ims = gen_sets.gen_sets("beta", plot_result=True)


#%% Results of 3 * 10 * 3 runs, please don't remove.
# mins = gen_sets.find_min_spread("factor")

# markov
# mins.mean(axis=0)
# array([-1.48273091, -1.61999238, -1.60832901])
# mins.std(axis=0)
# array([0.03983924, 0.03761347, 0.03058839])

# beta
# mins.mean(axis=0)
# array([-1.62771037, -1.61114015, -1.62307041])
# mins.std(axis=0)
# array([0.03643465, 0.03133686, 0.03066592])

# factor
# mins.mean(axis=0)
# array([-1.62368369, -1.61763135, -1.61375484])
# mins.std(axis=0)
# array([0.026696  , 0.02260477, 0.04707917])
