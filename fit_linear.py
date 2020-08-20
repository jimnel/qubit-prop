#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # use seaborn to visualize the weights with annotated value
plt.rcParams.update({'font.size': 13})

def fit_linear_reg(X, Y):
    """
    Input:
        X - The Collection of feature vectors
        Y - The Collection of target vectors
    Output:
        beta - The Linear Regression weights
    """
    cov = np.dot(X.T, X)  # covariance matrix 
    tmp = np.dot(x0.T, y)
    cov_inv = np.linalg.inv(cov)
    return np.dot(cov_inv, tmp)


x = np.load("bloch_evos.npy")
dt = 4  # time step

x0 = x[:, 0]  # initial states
y = x[:, dt]  # states at time dt
beta = fit_linear_reg(x0, y)


# view weights
plt.figure("weights")
plt.title("Regression Weights")
sns.heatmap(beta, annot=True, cmap="YlGnBu")


# save dt and the regression weights
fit_params = {"beta": beta,
              "dt": dt}

np.save("fit_params", fit_params)
plt.show()
