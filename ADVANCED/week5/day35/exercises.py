# Advanced Day 35 — Exercises

# Exercise 1 — Add Bidirectional LSTM
# Create BiLSTMClassifier using nn.LSTM(bidirectional=True).
# Concatenate both final hidden states before the output layer.
# Add it to the MODELS dict and re-run the benchmark.
# Does it outperform the uni-directional LSTM?
# TODO

# Exercise 2 — Learning rate sensitivity
# For each model, train with lr = [1e-4, 1e-3, 1e-2].
# Plot a 3x4 grid of loss curves (model vs lr).
# Which architectures are most sensitive to lr choice?
# TODO

# Exercise 3 — Sequence length sensitivity
# Train each model on SEQ_LEN = [4, 8, 16, 32].
# Report accuracy for each combination.
# Which architectures handle longer sequences better?
# TODO

# Exercise 4 — Noise robustness
# After training on clean data, test each model on noisy sequences:
# randomly replace 20% of tokens with random values.
# Measure accuracy drop. Which model is most robust?
# TODO

# Exercise 5 — FLOPs estimation
# Estimate the number of multiply-accumulate (MAC) operations for a
# single forward pass of each model.
# For Linear(in, out): MACs = in * out
# For RNN/LSTM hidden step: count the matrix multiplies
# Print MACs per model and rank by compute efficiency (acc/MACs).
# TODO
