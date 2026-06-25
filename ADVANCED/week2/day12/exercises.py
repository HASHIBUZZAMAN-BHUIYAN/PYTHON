# Advanced Day 12 — Exercises

# Exercise 1 — Augmentation pipeline
# Implement 5 augmentation functions for 8x8 images (all NumPy, no libraries):
# a) add_noise(img, sigma=0.05)
# b) brightness_jitter(img, factor_range=(0.8,1.2))
# c) horizontal_flip(img) — mirror left-right
# d) crop_and_pad(img, crop=1) — crop 1px and pad back to 8x8
# e) rotate_90(img) — rotate 90 degrees
# Apply each to a sample digit and plot.
# TODO

# Exercise 2 — Batch normalization study
# Train the same CNN with and without BatchNorm on digits.
# Plot train loss, train accuracy, test accuracy side by side.
# Why does BN help? Write a 2-sentence explanation in a comment.
# TODO

# Exercise 3 — Dropout placement
# Try placing dropout at different positions:
# a) After every conv layer
# b) Only before final linear layer (standard)
# c) No dropout
# Which placement gives best test accuracy?
# TODO

# Exercise 4 — Weight initialization
# Compare 3 initialization strategies for the CNN weights:
# a) Default (PyTorch Kaiming He)
# b) All zeros (bad!) — observe what happens
# c) All ones — observe
# Train each for 5 epochs. What do you notice?
# TODO

# Exercise 5 — Gradient clipping
# Train the CNN with a very high learning rate (0.1) — likely to diverge.
# Add torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# before optimizer.step(). Does clipping stabilize training?
# TODO
