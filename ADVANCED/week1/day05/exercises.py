# Advanced Day 05 — Exercises
import numpy as np
from sklearn.datasets import load_wine, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

np.random.seed(42)

# Exercise 1 — Multi-feature linear regression
# Generate 300 samples with 5 features.
# Train a linear regression. Print: R², RMSE, coefficients.
# Which feature has the largest (absolute) coefficient?
# TODO

# Exercise 2 — Logistic regression on Wine dataset
# Load wine dataset (sklearn built-in, 3 classes, 13 features).
# Train a logistic regression with scaling.
# Print accuracy, classification report, and confusion matrix.
# TODO

# Exercise 3 — Train/test split effects
# Using make_classification (200 samples, 5 features):
# Test 5 different test_size values: 0.1, 0.2, 0.3, 0.4, 0.5
# For each, train logistic regression and print train acc and test acc.
# What do you notice?
# TODO

# Exercise 4 — Manual polynomial features
# True relationship: y = 3x^2 - 2x + 1 + noise
# Linear regression on x alone will underfit.
# Add x^2 as a feature manually. Compare R² for both models.
x = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 3*x.flatten()**2 - 2*x.flatten() + 1 + np.random.normal(0, 1, 100)
# TODO

# Exercise 5 — Regularization intro
# from sklearn.linear_model import Ridge, Lasso
# Generate data with 10 features but only 3 are truly relevant.
# Compare: LinearRegression, Ridge(alpha=1), Ridge(alpha=100), Lasso(alpha=0.1)
# Print the coefficients for each and the test R².
# TODO
