# Advanced Day 06 — Exercises
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
np.random.seed(42)

# Exercise 1 — Breast cancer classification
# Load breast_cancer dataset (binary classification).
# Compare DT, RF, KNN, SVM. Print accuracy and AUC-ROC for each.
# TODO

# Exercise 2 — Learning curve
# For Random Forest, plot train accuracy vs test accuracy as
# n_estimators goes from 1 to 100 (step 10).
# This shows how performance improves with more trees.
# TODO

# Exercise 3 — Feature importance
# Train a RandomForest on the breast cancer dataset.
# Plot a bar chart of the top 10 most important features.
# TODO

# Exercise 4 — Cross-validation
# Use cross_val_score with cv=10 on the wine dataset.
# Compare mean accuracy ± std for DT, RF, KNN, SVM.
# TODO

# Exercise 5 — Overfitting demo
# Train a DecisionTreeClassifier with max_depth from 1 to 20.
# Plot training accuracy vs test accuracy vs max_depth.
# Identify the depth where test accuracy peaks (before overfitting).
# TODO
