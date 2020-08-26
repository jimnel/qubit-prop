#!/usr/bin/env python3
import qutip as qp
import numpy as np

h = np.array([0.7, 0.8, 0.2])  # hamiltonian parameters
lambda_param = 0.1  # linblaad term coupling strength
p = 3  # how many samples in our data-set
N_it = 200  # number of iterations

H = h[0]*qp.sigmax() + h[1]*qp.sigmay() + h[2]*qp.sigmaz()
T = np.linspace(0.0, 10.0, N_it)
dt = T[1]  # time-step size

bloch_vec_data = np.zeros((p, N_it, 3))

for i in range(3):
    psi0 = qp.rand_ket(2)
    result = qp.mesolve(H, psi0, T, [np.sqrt(lambda_param) * qp.sigmax()],
                        [qp.sigmax(), qp.sigmay(), qp.sigmaz()])
    bloch_vec_data[i] = np.array(result.expect).T

data = {"h": h, "lambda_param": lambda_param, "dt": dt,
        "bloch_vectors": bloch_vec_data}

np.save("DATA/data.npy", data)
