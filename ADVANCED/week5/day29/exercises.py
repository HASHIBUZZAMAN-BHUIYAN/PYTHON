# Day 29: RNN Fundamentals — Exercises
# Complete each TODO. Run the file to test your solutions.

import numpy as np
import torch
import torch.nn as nn

print("Day 29 Exercises — fill in each TODO\n")

# ─────────────────────────────────────────────
# Exercise 1: RNN Hidden State Shape
# ─────────────────────────────────────────────
print("Exercise 1: RNN output shapes")
# TODO: Create an nn.RNN with input_size=5, hidden_size=20, batch_first=True.
# Create a random input tensor of shape [4, 8, 5] (batch=4, seq_len=8, features=5).
# Run a forward pass and print the shapes of `output` and `h_n`.
# Expected: output=[4,8,20], h_n=[1,4,20]


# ─────────────────────────────────────────────
# Exercise 2: Manual Hidden State Update
# ─────────────────────────────────────────────
print("\nExercise 2: Manual hidden state update")
# TODO: Without using nn.RNN, manually implement one step of the RNN equation:
#   h_t = tanh(W_xh @ x_t + W_hh @ h_prev + b)
# Use numpy. Input x_t shape=(3,), h_prev shape=(4,).
# Initialize weights randomly (seed=0). Print the resulting h_t shape and values.


# ─────────────────────────────────────────────
# Exercise 3: Sequence to Scalar Prediction
# ─────────────────────────────────────────────
print("\nExercise 3: Sequence-to-scalar predictor")
# TODO: Build a small PyTorch model that:
#   - Takes a sequence of length 10, each element is a scalar (input_size=1)
#   - Uses nn.RNN(hidden_size=8, batch_first=True)
#   - Uses only the LAST hidden state (h_n) to predict a single output via nn.Linear(8, 1)
# Create synthetic data: 50 sequences where target = sum of sequence elements.
# Train for 30 epochs with MSELoss + Adam(lr=0.01). Print final loss.


# ─────────────────────────────────────────────
# Exercise 4: Gradient Clipping
# ─────────────────────────────────────────────
print("\nExercise 4: Gradient clipping")
# TODO: Using the model from Exercise 3 (or a fresh one), demonstrate gradient clipping.
# After loss.backward(), use torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Print the gradient norm BEFORE and AFTER clipping for one training step.
# Hint: compute norm before clipping manually with:
#   total_norm = sum(p.grad.norm()**2 for p in model.parameters())**0.5


# ─────────────────────────────────────────────
# Exercise 5: Multi-layer RNN
# ─────────────────────────────────────────────
print("\nExercise 5: Multi-layer RNN")
# TODO: Create a 2-layer nn.RNN (num_layers=2) with input_size=4, hidden_size=16.
# Pass a random input of shape [2, 6, 4] (batch=2, seq=6, features=4).
# Print the shape of h_n and explain in a comment why h_n has shape [2, 2, 16]
# (i.e., what each dimension represents).


print("\nAll exercises attempted!")
