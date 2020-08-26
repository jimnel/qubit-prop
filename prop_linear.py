#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})

data = np.load("DATA/data.npy", allow_pickle=True).item()
x = data['bloch_vectors']

fit_params = np.load("DATA/fit_params.npy")

Ts = np.arange(data['bloch_vectors'].shape[1]) * data['dt']


# we will now test our linear model by using autoregression to see
# how far into the future we can predict
preds = np.zeros(shape=x.shape)
preds[:, 0] = x[:, 0]  # initialize to x0

# autoregression (feed predictions back into the model)
for i in range(x.shape[1]-1):
    preds[:, i+1] = np.dot(preds[:, i], fit_params)

# Figure 1: Mean error
trace_error = 0.5*np.sqrt(((preds - x)**2).sum(-1))
mean_trace_error = trace_error.mean(0)

plt.figure("prop_error")
plt.plot(Ts, mean_trace_error, c='k')
plt.plot(Ts, trace_error.T, c='C0', lw=0.4)
plt.xlabel("Time")
plt.ylabel("Trace Error")
plt.tight_layout()


# Figure 2: Propagation
fig, axs = plt.subplots(nrows=3, num="prop example", sharex="col")
labels = ['x', 'y', 'z']
for i in range(3):
    axs[i].plot(Ts, x[0, :, i], label="True", lw=2)
    axs[i].plot(Ts, preds[0, :, i], label="Pred", ls="--", lw=2)
    axs[i].set_ylabel(r"$\langle \sigma_%s \rangle$" % labels[i])
axs[-1].set_xlabel("Time")
axs[-1].legend()
plt.tight_layout()

plt.show()


