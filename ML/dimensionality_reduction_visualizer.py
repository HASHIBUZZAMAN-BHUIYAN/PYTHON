"""
What it does: Reduces the 64-feature sklearn digits dataset to 2D using PCA and t-SNE.
What it teaches: PCA explained variance, dimensionality reduction pipeline,
                 t-SNE non-linear manifold learning, runtime comparison.
Category: DIMENSIONALITY REDUCTION
RAM estimate: < 200 MB
Time estimate: 15-30 seconds (t-SNE is the slow step)
"""

import os
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("ML/outputs", exist_ok=True)

digits = load_digits()
X = digits.data          # (1797, 64)
y = digits.target        # 0-9
print(f"Dataset shape: {X.shape}, classes: {np.unique(y)}")

t0 = time.time()
pca50 = PCA(n_components=50, random_state=42)
X_pca50 = pca50.fit_transform(X)
ev_50 = pca50.explained_variance_ratio_.sum()
pca_50d_time = time.time() - t0
print(f"\nPCA 50D - explained variance: {ev_50:.2%} (took {pca_50d_time:.2f}s)")

t0 = time.time()
pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X)
ev_2 = pca2.explained_variance_ratio_
pca_2d_time = time.time() - t0
print(f"PCA  2D - PC1={ev_2[0]:.2%}, PC2={ev_2[1]:.2%}, "
      f"total={ev_2.sum():.2%} (took {pca_2d_time:.2f}s)")

# Cumulative explained variance
cum_ev = np.cumsum(pca50.explained_variance_ratio_)
for thresh in [0.80, 0.90, 0.95]:
    n_needed = np.searchsorted(cum_ev, thresh) + 1
    print(f"  Components needed for {thresh:.0%} variance: {n_needed}")

print("\nRunning t-SNE on PCA-50D data (this may take ~20s)...")
t0 = time.time()
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
            random_state=42, init="pca", learning_rate="auto")
X_tsne = tsne.fit_transform(X_pca50)
tsne_time = time.time() - t0
print(f"t-SNE 2D done (took {tsne_time:.2f}s)")

print("\nRuntime Comparison")
print("-" * 40)
print(f"  PCA  50D:  {pca_50d_time:.3f}s")
print(f"  PCA   2D:  {pca_2d_time:.3f}s")
print(f"  t-SNE 2D:  {tsne_time:.3f}s")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
palette = plt.cm.tab10.colors

for ax, X_2d, title in zip(
        axes,
        [X_pca2, X_tsne],
        ["PCA 2D", "t-SNE 2D (on PCA-50D)"]):
    for digit in range(10):
        mask = y == digit
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=palette[digit], s=10, alpha=0.7, label=str(digit))
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(title="Digit", fontsize=7, markerscale=2,
              loc="best", ncol=2)

plt.suptitle("Dimensionality Reduction - Digits Dataset (1797 samples, 64 features)",
             fontsize=11)
plt.tight_layout()
plt.savefig("ML/outputs/dim_reduction.png", dpi=100)
plt.close()
print("Plot saved -> ML/outputs/dim_reduction.png")
