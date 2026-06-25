# Advanced Day 13 — Exercises

# Exercise 1 — Fine-tuning
# Start with the frozen MobileNetV2 from the lesson.
# After training the head for 5 epochs, UNFREEZE the last 5 layers
# and continue training with lr=1e-4.
# Compare final accuracy with head-only training.
# TODO

# Exercise 2 — Feature extraction as preprocessing
# Instead of running backbone on every batch every epoch,
# run the backbone ONCE on all data, save the features, then
# train a simple linear layer on those features.
# This is much faster for CPU. Compare speed vs per-epoch extraction.
# TODO

# Exercise 3 — Different backbone sizes
# Compare feature extraction accuracy with:
# a) MobileNetV2 (default)
# b) A tiny custom CNN (3 conv layers, ~50K params)
# c) ResNet18 (if you have memory — ~11M params, ~450 MB)
# For each, print: param count, training time, final test accuracy.
# TODO

# Exercise 4 — Class activation maps (CAM)
# After training the frozen backbone, implement a simple version of
# Grad-CAM or just Class Activation Map (CAM) to show which pixels
# the model focuses on for each predicted class.
# Visualize on a few test images.
# TODO

# Exercise 5 — Domain shift
# Train the model on shapes with a light background.
# Then test it on shapes with a dark background (invert the images: img = 1 - img).
# How much does accuracy drop? How do you fix it?
# TODO
