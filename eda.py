#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13})

data = np.load("DATA/data.npy", allow_pickle=True).item()

T = np.arange(data['bloch_vectors'].shape[1]) * data['dt']

plt.figure("vector norm")
r = np.sqrt((data['bloch_vectors'][:10]**2).sum(-1))
plt.plot(T, r.T, c="C0", lw=2)
plt.xlabel("Time")
plt.ylabel("Block Vector Magnitude")
plt.tight_layout()

plt.show()

