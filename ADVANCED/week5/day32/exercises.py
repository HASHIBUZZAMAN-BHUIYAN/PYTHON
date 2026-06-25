# Advanced Day 32 — Exercises

# Exercise 1 — Mode collapse demo
# Make the discriminator much stronger by using lr_D=1e-3, lr_G=1e-5.
# Train for 200 epochs. Plot the generated distribution.
# Does the generator collapse to generating only a few modes?
# TODO

# Exercise 2 — Wasserstein loss
# Replace BCE with Wasserstein (WGAN) loss:
#   D loss: -mean(D(real)) + mean(D(fake))
#   G loss: -mean(D(fake))
# Remove the Sigmoid from D. Clip D weights after each step: param.data.clamp(-0.01, 0.01)
# Does training become more stable?
# TODO

# Exercise 3 — Conditional GAN
# Add a class label (0-3 for each Gaussian mode) as input to both G and D.
# G(z, label) → sample from the conditional distribution for that mode.
# Sample one point per mode and verify each lands near the right cluster.
# TODO

# Exercise 4 — Diversity metric
# After training, sample 512 points from G.
# Compute the pairwise distance standard deviation as a diversity proxy.
# High diversity = generator captures multiple modes.
# Compare diversity of G vs real data.
# TODO

# Exercise 5 — FID-style metric
# Compute the feature statistics (mean, covariance) of real and generated samples.
# Compute Frechet Distance: FD = ||mu_r - mu_g||^2 + Tr(Σ_r + Σ_g - 2*(Σ_r @ Σ_g)^0.5)
# Use scipy.linalg.sqrtm for the matrix square root.
# TODO
