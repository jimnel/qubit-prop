#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13})

x = np.load("bloch_evos.npy")

plt.figure("vector norm")
r = np.sqrt((x[:10]**2).sum(-1))
plt.plot(r.T, c="C0", lw=0.4)
plt.plot(r.mean(0), c="k")
plt.xlabel("Iterations")
plt.ylabel("Block Vector Magnitude")
plt.tight_layout()

plt.show()

