# Advanced Day 31 — Exercises

# Exercise 1 — Latent dimension effect
# Train autoencoders with latent_dim = [1, 2, 4, 8] on the same 8D data.
# Plot reconstruction loss vs latent dimension on a bar chart.
# At what latent dim does loss stop decreasing significantly?
# TODO

# Exercise 2 — Deeper encoder
# Add one more hidden layer to both encoder and decoder (8→16→8→latent).
# Compare final reconstruction loss vs the shallow version from lesson.
# TODO

# Exercise 3 — Denoising with different noise levels
# Train denoising AEs with noise_std = [0.1, 0.5, 1.0, 2.0].
# For each, compute the denoising improvement ratio.
# Plot improvement vs noise level.
# TODO

# Exercise 4 — Anomaly threshold sweep
# Vary the anomaly threshold from the 50th to 99th percentile of normal errors.
# For each threshold, compute: true positive rate (anomalies caught) and
# false positive rate (normals flagged). Plot ROC-style curve.
# TODO

# Exercise 5 — PCA vs Autoencoder comparison
# Compress the 8D data to 2D using both PCA and the AE encoder.
# Scatter plot both side-by-side colored by sample index.
# Which gives cleaner structure? Why might they differ?
# TODO
