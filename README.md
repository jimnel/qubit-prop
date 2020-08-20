# qubit-prop
Learn a simple linear model from some Bloch vector dynamics

The file 'bloch_evos.npy' contains 10 Bloch vector trajectories each with different initial conditions. 

The evolutions are non-unitary and the Bloch vectors increase in entropy over the course of the evolution. This is shown in the 'eda.npy' code where the magnitude of the Bloch vectors is shown over time (Figure 'vector_norm.png').

'fit_linear.py' fits the data to a linear model and plots the weights (Figure 'weights.png').

'prop_linear.py' shows how the model can predict the evolution of the states by using autoregression. Witht the results in Figures 'prop_error.png' and 'prop_example.png'.
