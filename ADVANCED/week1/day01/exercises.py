# Advanced Day 01 — Exercises

import numpy as np
np.random.seed(0)

# Exercise 1 — Array creation
# Create a 5x5 matrix where the border is 1 and the interior is 0.
# Expected:
# [[1 1 1 1 1]
#  [1 0 0 0 1]
#  [1 0 0 0 1]
#  [1 0 0 0 1]
#  [1 1 1 1 1]]
# Use np.ones and slicing. No explicit loops.
# TODO

# Exercise 2 — Vectorized normalization
# Normalize a random (100,) array to have mean 0 and std 1.
# (x - mean) / std — do it without any Python loops.
# Verify: print mean and std of result (should be ~0 and ~1).
# TODO

# Exercise 3 — Broadcasting puzzle
# Without loops, compute the outer product of [1,2,3] and [4,5,6,7].
# Expected shape: (3,4). Do NOT use np.outer — use broadcasting with reshape.
# TODO

# Exercise 4 — Matrix statistics
# Create a 6x4 random integer matrix (values 1-100).
# Print: row with highest mean, column with lowest std, overall median.
# TODO

# Exercise 5 — Linear regression via NumPy
# Given x and y below, find the best-fit line y = mx + b using:
# the normal equation: [m, b] = (X^T X)^-1 X^T y
# where X is the design matrix [[x1,1],[x2,1],...].
# Print m, b, and the predicted y values.
x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
y = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 14.2, 15.9])
# TODO
