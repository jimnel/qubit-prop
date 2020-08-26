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

data = np.load("DATA/data.npy", allow_pickle=True).item()
x = data['bloch_vectors']

x0 = x[:, 0]  # initial states
y = x[:, 1]  # states at first time step
beta = fit_linear_reg(x0, y)

print("The weights are:\n", beta)

# view weights
plt.figure("weights")
plt.title("Regression Weights")
sns.heatmap(beta, annot=True, cmap="YlGnBu")


# save the regression weights

np.save("DATA/fit_params", beta)
plt.show()
