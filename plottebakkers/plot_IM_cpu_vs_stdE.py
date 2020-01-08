import pickle
import matplotlib.pyplot as plt
import numpy as np

neighbourhoods = np.arange(1, 6)

plt.rcParams.update({'font.size': 13})
energies = pickle.load(open("../data/picklejar/IM_energies.pkl", "rb"))
cpu_times = pickle.load(open("../data/picklejar/IM_cpu_times.pkl", "rb"))

dx = 0.5
x = cpu_times.mean(axis=1)
y = energies.std(axis=1)
plt.title("Averaged CPU runtime vs std in energy minima \n"
          " for different neighbourhoods n.")
plt.scatter(x, y, c='black')
plt.xlabel("CPU runtime (s)")
plt.ylabel(r"$\sigma (E_\mathrm{min})$ over 20 iterations")

for i, n in enumerate(neighbourhoods):
    if n == 3:
        plt.annotate('n = {}'.format(n), (x[i] - dx * 2, y[i] + dx / 3))
    elif n == 5:
        plt.annotate('n = {}'.format(n), (x[i] - dx * 4, y[i] + dx / 3))
    else:
        plt.annotate('n = {}'.format(n), (x[i] + dx, y[i] - dx / 3))

plt.savefig('../data/IM_cpu_vs_stdE.png')
