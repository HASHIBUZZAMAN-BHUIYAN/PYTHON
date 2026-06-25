# Advanced Day 08 — Exercises

import numpy as np
np.random.seed(42)

# Exercise 1 — Activation functions
# Plot sigmoid, tanh, and ReLU on x in [-5, 5].
# Also plot their derivatives on a separate subplot row.
# TODO

# Exercise 2 — Single perceptron (AND gate)
# Implement a single perceptron (1 layer, sigmoid).
# Train it to learn the AND function:
# (0,0)→0, (0,1)→0, (1,0)→0, (1,1)→1
# Print weights, bias, predictions after 1000 epochs.
# TODO

# Exercise 3 — Gradient check
# Verify backprop numerically for the TwoLayerNet from lesson.py.
# For a small batch (4 samples), compute the numerical gradient
# for W1[0,0] using:   (loss(w+eps) - loss(w-eps)) / (2*eps)
# Compare to the analytical gradient. They should be very close.
# TODO

# Exercise 4 — Mini-batch gradient descent
# Copy the TwoLayerNet and modify the train() method to use mini-batches.
# Add a batch_size parameter (default 32).
# Train on the circles dataset with batch_size=16, 32, 64.
# Compare convergence speed (loss after 500 epochs).
# TODO

# Exercise 5 — Regression with NN (no sigmoid output)
# Create a dataset: y = 0.5*x^2 + noise.
# Build a network with 1 input, 8 hidden neurons (ReLU), 1 linear output.
# Train and plot predictions vs true curve.
# TODO
