import pickle
import matplotlib.pyplot as plt
from matplotlib import ticker

with open("../data/IM_sa_fer.pkl", "rb") as f:
    cnt_fer = pickle.load(f)
    betas_fer = pickle.load(f)
    me_fer = pickle.load(f) / 100
    stde_fer = pickle.load(f) / 100

with open("../data/IM_sa_fr.pkl", "rb") as f:
    cnt_fr = pickle.load(f)
    betas_fr = pickle.load(f)
    me_fr = pickle.load(f) / 100
    stde_fr = pickle.load(f) / 100

temperature_fer = 1 / betas_fer
temperature_fr = 1 / betas_fr

#%%
plt.rcParams.update({'font.size': 17})
fig, ax = plt.subplots(2, 2, sharex='all', figsize=(10, 10))
for ax0 in ax[0]:
    ax0.hlines(y=0, xmin=temperature_fer.min() - 10,
               xmax=temperature_fer.max() + 100,
               colors='grey', linestyles='dotted')

ax[0][0].set_xscale('log')  # Set x scale of entire plot
ax[0][0].set_xlim(temperature_fr.min(), temperature_fr.max())
ax[0][0].set_ylim(-7.2, 0.2)
ax[0][1].set_ylim(-7.2, 0.2)

ax[0][0].set_ylabel("(Mean) energy")
ax[0][0].plot(temperature_fer, me_fer, c='black')
ax[0][1].plot(temperature_fr, me_fr, c='black')
# ax[0][0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%f'))

ax[1][0].set_ylabel(r"$\sigma(E)$")
ax[1][0].plot(temperature_fer, stde_fer, c='black')
ax[1][1].plot(temperature_fr, stde_fr, c='black')

# Set x labels underneath and titles above
ax[0][0].set_title("Ferro-magnetic")
ax[0][1].set_title("Frustrated")
ax[1][0].set_xlabel("Temperature")
ax[1][1].set_xlabel("Temperature")

plt.savefig("../data/McKayFig3111.png")
