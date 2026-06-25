# Advanced Day 07 — Exercises
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
np.random.seed(42)

# Use a small subset of digits (500 samples, all 10 classes)
digits = load_digits()
idx = np.random.choice(len(digits.data), 500, replace=False)
X, y = digits.data[idx], digits.target[idx]

# Exercise 1 — Multi-class metrics
# Train a RandomForest on the digits subset (70/30 split).
# Print per-class precision, recall, F1.
# Print weighted average F1.
# TODO

# Exercise 2 — Threshold tuning
# Logistic Regression for binary: classify digit "5" vs rest.
# Default threshold = 0.5. Try thresholds 0.3, 0.4, 0.5, 0.6, 0.7.
# Print precision and recall at each threshold.
# Which threshold maximizes recall? Which maximizes precision?
# TODO

# Exercise 3 — Imbalanced classes
# Create an imbalanced binary problem: 90% class 0, 10% class 1.
# Train logistic regression. Compare accuracy vs F1.
# Show why accuracy can be misleading for imbalanced data.
# Also try class_weight="balanced" — does it help?
# TODO

# Exercise 4 — Pipeline with grid search
# Build a Pipeline: StandardScaler + RandomForest.
# Use GridSearchCV to tune: n_estimators (10,50), max_depth (3,5,None).
# Print best params, best CV score, test score.
# TODO

# Exercise 5 — Validation curve
# Plot validation curve for RandomForest: vary max_depth from 1 to 15.
# Show train score and CV score vs max_depth. Identify sweet spot.
# TODO
