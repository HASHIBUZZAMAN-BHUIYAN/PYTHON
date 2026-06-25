# Advanced Day 04 — Exercises
import numpy as np
from scipy import stats
np.random.seed(0)

# Exercise 1 — Describe a dataset
# Compute all standard descriptive stats for the array below.
# Print them in a formatted table. Include outlier detection:
# flag values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] as outliers.
data = [4,7,13,2,1,8,6,3,14,99,5,9,2,7,4,3,8,6,5,11]
# TODO

# Exercise 2 — Distributions
# a) Sample 10000 values from N(mu=50, sigma=10).
#    What fraction fall within 1, 2, 3 standard deviations? (68-95-99.7 rule)
# b) If a factory produces 1000 widgets and each fails independently with p=0.02,
#    what is the probability that at most 30 fail? Use Poisson approximation.
# TODO

# Exercise 3 — Correlation analysis
# Generate three variables: x (uniform 0-100), y (linear in x + noise), z (random).
# Compute all pairwise Pearson correlations.
# Which pair is most correlated? Plot a scatter matrix (3x3 subplots).
# TODO

# Exercise 4 — A/B test simulation
# Website A converts at 5%, website B at 7%.
# Simulate 500 visitors to each. Run a chi-square or proportion z-test.
# Print result: is the difference statistically significant?
# Repeat 100 times and print how often we correctly reject H0.
# TODO

# Exercise 5 — Bootstrap confidence interval
# Given the sample below, estimate the 95% CI for the mean using bootstrapping
# (resample with replacement 10000 times, take 2.5th and 97.5th percentile).
sample = np.array([23, 31, 18, 25, 29, 22, 35, 27, 19, 30, 24, 28, 21, 33, 26])
# TODO
