# Advanced Day 26 — Exercises

# Exercise 1 — Add a 5th category
# Add "Finance" with 8 sample sentences to the CORPUS.
# Re-train TF-IDF+LR and measure accuracy change.
# TODO

# Exercise 2 — Confusion matrix heatmap
# Fit TF-IDF+LR on the full corpus (train/test split 80/20).
# Plot a labeled confusion matrix heatmap using matplotlib.
# TODO

# Exercise 3 — Feature importance per class
# For the fitted LogisticRegression, find the top 5 words
# most predictive of each class (highest absolute coefficient).
# Print as a table.
# TODO

# Exercise 4 — Misclassification analysis
# Find all misclassified samples from the test set.
# For each misclassification print: true label, predicted label, and the text.
# What patterns do you see? (e.g., Sport/Politics overlap)
# TODO

# Exercise 5 — Threshold-based "Other" class
# After predicting with TF-IDF+LR, use predict_proba to get confidence.
# If max confidence < 0.4, classify as "UNSURE" instead.
# Report how many samples were classified as UNSURE.
# TODO
