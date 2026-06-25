# Day 30: LSTM & GRU — Exercises
# Complete each TODO. Run the file to test your solutions.

import torch
import torch.nn as nn
import numpy as np

print("Day 30 Exercises — fill in each TODO\n")

# ─────────────────────────────────────────────
# Exercise 1: LSTM vs GRU Parameter Count
# ─────────────────────────────────────────────
print("Exercise 1: LSTM vs GRU parameter count")
# TODO: Create nn.LSTM(input_size=10, hidden_size=32) and
# nn.GRU(input_size=10, hidden_size=32).
# Print the total parameter count for each.
# Manually verify: LSTM = 4 * (input+hidden+1) * hidden,
#                  GRU  = 3 * (input+hidden+1) * hidden
# (the +1 is for bias — there are actually 2 bias vectors per gate in PyTorch)


# ─────────────────────────────────────────────
# Exercise 2: Bidirectional LSTM
# ─────────────────────────────────────────────
print("\nExercise 2: Bidirectional LSTM")
# TODO: Create a bidirectional LSTM: nn.LSTM(..., bidirectional=True).
# input_size=6, hidden_size=12, batch_first=True.
# Pass input of shape [2, 8, 6]. Print output shape and explain in a comment
# why output has size 24 (not 12) on the last dimension.


# ─────────────────────────────────────────────
# Exercise 3: LSTM for Sequence Classification
# ─────────────────────────────────────────────
print("\nExercise 3: LSTM sequence classifier")
# TODO: Build a model that:
#   - Input: [batch, 15, 1] (scalar sequence of length 15)
#   - LSTM hidden_size=16
#   - Use the LAST timestep's output (out[:, -1, :]) to classify into 2 classes
#   - Add nn.Linear(16, 2) on top
# Create 80 random sequences, labels = 1 if sequence mean > 0 else 0.
# Train for 20 epochs, print final accuracy.


# ─────────────────────────────────────────────
# Exercise 4: GRU with Dropout
# ─────────────────────────────────────────────
print("\nExercise 4: GRU with dropout")
# TODO: Create an nn.GRU with input_size=8, hidden_size=24, num_layers=2, dropout=0.3.
# Note: dropout only applies BETWEEN layers (not on last layer), so num_layers must be >= 2.
# Print total parameters. Run a forward pass with input [3, 12, 8].
# Print output shape. Observe that output shape is the same as without dropout.


# ─────────────────────────────────────────────
# Exercise 5: Packing for Variable-Length Sequences
# ─────────────────────────────────────────────
print("\nExercise 5: Packed sequences")
# TODO: Demonstrate using pack_padded_sequence + pad_packed_sequence for
# variable-length inputs. Create 3 sequences of lengths [6, 4, 3], padded to length 6.
# Use: torch.nn.utils.rnn.pack_padded_sequence and pad_packed_sequence.
# Pass through nn.LSTM(input_size=4, hidden_size=8, batch_first=True).
# Print the output shape before and after unpacking.
# Hint: sequences must be sorted by length (descending).


print("\nAll exercises attempted!")
