# Advanced Day 09 — Exercises
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
np.random.seed(42); torch.manual_seed(42)

# Exercise 1 — Tensor operations
# a) Create a 5x5 matrix of random floats, compute its inverse using torch.linalg.
# b) Create two 3D tensors (batch_size=4, rows=3, cols=2), compute bmm (batch matmul).
# c) Stack 4 random (2,2) tensors into a (4,2,2) tensor using torch.stack.
# TODO

# Exercise 2 — Custom nn.Module
# Build a "Highway Network" block:
# class HighwayBlock(nn.Module):
#   H = Linear(n,n) + ReLU  (transform gate)
#   T = Linear(n,n) + Sigmoid (carry gate)
#   output = H(x)*T(x) + x*(1-T(x))
# Create a network: Linear(10,64) → 3 HighwayBlock(64) → Linear(64,1)
# Count parameters. Test with a (16,10) batch.
# TODO

# Exercise 3 — Different optimizers
# Train a simple net on make_moons (sklearn) with:
# SGD(lr=0.1), Adam(lr=0.01), RMSprop(lr=0.01)
# Plot train loss over 200 epochs for each. Which converges fastest?
# TODO

# Exercise 4 — Saving and loading a model
# Train a tiny net for 50 epochs. Save it with torch.save().
# Load it back, verify it produces the same predictions.
# TODO

# Exercise 5 — Batch normalization
# Build two identical networks on digits data, one with BatchNorm1d, one without.
# Compare convergence (loss after each epoch) over 10 epochs.
# TODO
