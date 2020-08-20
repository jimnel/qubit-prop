# qubit-prop
Learn a simple linear model from some Bloch vector dynamics

The file 'bloch_evos.npy' contains 10 Bloch vector trajectories each with diffent initial conditions. 

The evolutions are non-unitary and the Bloch vectors increase in entropy over the course of the evolution. This is shown in the 'eda.npy' code where the magnitude of the Bloch vectors is shown over time.

'fit_linear.py' fits the data to a linear model and plots the weights.

'prop_linear.py' shows how the model can predict the evolution of the states by using autoregression.
