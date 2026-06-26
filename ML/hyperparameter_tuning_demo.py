"""
Hyperparameter Tuning Demo
===========================
Category: MODEL SELECTION & HYPERPARAMETER TUNING

What it does:
  Compares GridSearchCV (exhaustive) vs RandomizedSearchCV (random sampling)
  on a RandomForestClassifier tuning task. Shows:
    - How many model fits each approach performs
    - Which finds the better hyperparameters
    - Time tradeoff between exhaustive and random search
    - How to read the results heatmap

What it teaches:
  - GridSearchCV: tries every combination in the grid (guaranteed to find best)
  - RandomizedSearchCV: samples n_iter combinations randomly (faster, good approximation)
  - Cross-validation inside search (no data leakage)
  - How to choose between the two approaches

How to run:
  python ML\hyperparameter_tuning_demo.py   (from PYTHON\ folder)

Estimated RAM: ~100MB | Time: ~30s
Dataset: synthetic classification (sklearn.make_classification, 500 samples)
"""

import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

os.makedirs("ML/outputs", exist_ok=True)

print("Hyperparameter Tuning Demo")
print("=" * 55)

rng = np.random.RandomState(42)
X, y = make_classification(
    n_samples=500, n_features=8, n_informative=5,
    n_redundant=2, random_state=42
)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, class balance: {y.mean():.2f}")

base_model  = RandomForestClassifier(random_state=42)
base_cv     = cross_val_score(base_model, X, y, cv=5, scoring="accuracy")
print(f"\nBaseline (default params): {base_cv.mean():.4f} +/- {base_cv.std():.4f}")

param_grid = {
    "n_estimators":    [50, 100, 200],
    "max_depth":       [3, 5, None],
    "min_samples_leaf":[1, 2, 4],
    "max_features":    ["sqrt", "log2"],
}
# Grid has 3 * 3 * 3 * 2 = 54 combinations, each with 5-fold CV = 270 fits
total_grid_fits = 3 * 3 * 3 * 2 * 5
n_iter_random   = 15

print(f"\nSearch space: {3*3*3*2} combinations")
print(f"  GridSearchCV   : all {3*3*3*2} combos x 5-fold = {total_grid_fits} fits")
print(f"  RandomizedSearchCV: {n_iter_random} combos x 5-fold = {n_iter_random*5} fits")

print("\nRunning GridSearchCV (exhaustive)...")
t0 = time.time()
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5, scoring="accuracy", n_jobs=-1, refit=True
)
grid_search.fit(X, y)
grid_time = time.time() - t0

print(f"  Done in {grid_time:.1f}s")
print(f"  Best score    : {grid_search.best_score_:.4f}")
print(f"  Best params   : {grid_search.best_params_}")

print(f"\nRunning RandomizedSearchCV (n_iter={n_iter_random})...")
t0 = time.time()
rand_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    n_iter=n_iter_random, cv=5, scoring="accuracy",
    n_jobs=-1, refit=True, random_state=42
)
rand_search.fit(X, y)
rand_time = time.time() - t0

print(f"  Done in {rand_time:.1f}s")
print(f"  Best score    : {rand_search.best_score_:.4f}")
print(f"  Best params   : {rand_search.best_params_}")

print("\n" + "=" * 55)
print("  COMPARISON SUMMARY")
print("-" * 55)
print(f"  {'Method':<25} {'Best CV Acc':>11}  {'Time':>8}  {'Fits':>6}")
print("-" * 55)
print(f"  {'Baseline (defaults)':<25} {base_cv.mean():>11.4f}  {'N/A':>8}  {'N/A':>6}")
print(f"  {'GridSearchCV':<25} {grid_search.best_score_:>11.4f}  {grid_time:>7.1f}s  {total_grid_fits:>6}")
print(f"  {'RandomizedSearchCV':<25} {rand_search.best_score_:>11.4f}  {rand_time:>7.1f}s  {n_iter_random*5:>6}")
print("-" * 55)

gain_grid = (grid_search.best_score_ - base_cv.mean()) * 100
gain_rand = (rand_search.best_score_ - base_cv.mean()) * 100
print(f"\n  Gain over baseline:")
print(f"    GridSearchCV      : +{gain_grid:.2f}%")
print(f"    RandomizedSearchCV: +{gain_rand:.2f}%")

speedup = grid_time / max(rand_time, 0.001)
print(f"\n  GridSearch is {speedup:.1f}x slower than RandomizedSearch")
print(f"  Score gap: {abs(grid_search.best_score_ - rand_search.best_score_)*100:.2f}% difference")

# Pivot: n_estimators vs max_depth, averaged over other params
import pandas as pd
results_df = pd.DataFrame(grid_search.cv_results_)

heatmap_data = {}
for _, row in results_df.iterrows():
    key = (row["param_n_estimators"], str(row["param_max_depth"]))
    if key not in heatmap_data or row["mean_test_score"] > heatmap_data[key]:
        heatmap_data[key] = row["mean_test_score"]

n_est_vals  = sorted({k[0] for k in heatmap_data})
depth_vals  = ["3", "5", "None"]
grid_matrix = np.zeros((len(depth_vals), len(n_est_vals)))
for i, d in enumerate(depth_vals):
    for j, n in enumerate(n_est_vals):
        grid_matrix[i, j] = heatmap_data.get((n, d), 0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

im = axes[0].imshow(grid_matrix, cmap="YlOrRd", aspect="auto",
                    vmin=grid_matrix.min()-0.01, vmax=grid_matrix.max()+0.01)
axes[0].set_xticks(range(len(n_est_vals))); axes[0].set_xticklabels(n_est_vals)
axes[0].set_yticks(range(len(depth_vals)));  axes[0].set_yticklabels(depth_vals)
axes[0].set_xlabel("n_estimators"); axes[0].set_ylabel("max_depth")
axes[0].set_title("GridSearch: Best CV Accuracy\n(n_estimators vs max_depth)")
for i in range(len(depth_vals)):
    for j in range(len(n_est_vals)):
        axes[0].text(j, i, f"{grid_matrix[i,j]:.3f}", ha="center", va="center",
                     fontsize=9, color="black")
plt.colorbar(im, ax=axes[0])

methods = ["Baseline", "GridSearch", "RandomSearch"]
scores  = [base_cv.mean(), grid_search.best_score_, rand_search.best_score_]
errs    = [base_cv.std(), 0, 0]
colors  = ["steelblue", "tomato", "seagreen"]
bars = axes[1].bar(methods, scores, yerr=errs, capsize=5, color=colors, alpha=0.85)
axes[1].set_ylim(min(scores) - 0.05, 1.0)
axes[1].set_ylabel("5-Fold CV Accuracy")
axes[1].set_title("Method Comparison")
axes[1].grid(axis="y", alpha=0.3)
for bar, score in zip(bars, scores):
    axes[1].text(bar.get_x() + bar.get_width()/2, score + 0.005,
                 f"{score:.4f}", ha="center", va="bottom", fontsize=9)

timing_txt = f"Grid: {grid_time:.1f}s\nRand: {rand_time:.1f}s"
axes[1].text(0.98, 0.05, timing_txt, transform=axes[1].transAxes,
             ha="right", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.suptitle("Hyperparameter Tuning: GridSearch vs RandomizedSearch", fontsize=11)
plt.tight_layout()
outpath = "ML/outputs/hyperparameter_tuning.png"
plt.savefig(outpath, dpi=90)
plt.close()
print(f"\nPlot saved -> {outpath}")
print("\n[DONE] hyperparameter_tuning_demo.py complete")
