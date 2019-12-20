import pickle
import matplotlib.pyplot as plt

with open("../data/IM_sa_fer.pkl", "rb") as f:
    cnt_fer = pickle.load(f)
    betas_fer = pickle.load(f)
    me_fer = pickle.load(f)
    stde_fer = pickle.load(f)

with open("../data/IM_sa_fer.pkl", "rb") as f:
    cnt_fr = pickle.load(f)
    betas_fr = pickle.load(f)
    me_fr = pickle.load(f)
    stde_fr = pickle.load(f)

plt.rcParams.update({'font.size': 13})
temperature_fer = 1 / betas_fer
temperature_fr = 1 / betas_fr

fig, ax = plt.subplots(2, 2, sharex='all', sharey='row', figsize=(10, 10))
ax[0][0].set_xscale('log')  # Set x scale of entire plot
ax[0][0].set_xlim(temperature_fr.min(), temperature_fr.max())

for ax0 in ax[0]:
    ax0.hlines(y=0, xmin=temperature_fer.min() - 10,
               xmax=temperature_fer.max() + 100,
               colors='grey', linestyles='dotted')
ax[0][0].set_ylabel("(Mean) energy")
ax[0][0].plot(temperature_fer, me_fer)
ax[0][1].plot(temperature_fr, me_fr)

ax[1][0].set_ylabel(r"$\sigma(E)$")
ax[1][0].plot(temperature_fer, stde_fer)
ax[1][1].plot(temperature_fr, stde_fr)

# Set x labels underneath
ax[1][0].set_xlabel("Temperature")
ax[1][1].set_xlabel("Temperature")

fig.show()