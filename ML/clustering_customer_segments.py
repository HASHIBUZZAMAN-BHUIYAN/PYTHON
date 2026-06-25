"""
What it does: Groups synthetic customers into segments using KMeans and DBSCAN.
What it teaches: Unsupervised clustering, KMeans, DBSCAN, silhouette score,
                 PCA for 2D visualization of multi-dimensional clusters.
Category: CLUSTERING (UNSUPERVISED)
RAM estimate: < 100 MB
Time estimate: < 5 seconds
"""

import os
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("ML/outputs", exist_ok=True)

# --- Generate synthetic customer data with 4 natural clusters ---
np.random.seed(42)
centers = [
    [2000, 5,  30],   # Low spend, low frequency, small basket
    [8000, 20, 80],   # High spend, high frequency, big basket
    [3000, 15, 60],   # Mid spend, high frequency, big basket
    [6000, 3,  100],  # High spend, low frequency, very big basket
]
X_raw, y_true = make_blobs(n_samples=500, centers=centers,
                           cluster_std=[300, 500, 400, 600],
                           random_state=42)
# Label columns conceptually: annual_spend, visit_frequency, avg_basket_size

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# --- KMeans (k=4) ---
km = KMeans(n_clusters=4, random_state=42, n_init=10)
km_labels = km.fit_predict(X)
km_sil = silhouette_score(X, km_labels)

print("KMeans (k=4) Results")
print("-" * 40)
for k in range(4):
    count = (km_labels == k).sum()
    centroid_raw = scaler.inverse_transform(km.cluster_centers_[k].reshape(1, -1))[0]
    print(f"  Cluster {k}: {count} samples | "
          f"spend={centroid_raw[0]:,.0f} freq={centroid_raw[1]:.1f} basket={centroid_raw[2]:.1f}")
print(f"  Silhouette score: {km_sil:.4f}")

# --- DBSCAN ---
db = DBSCAN(eps=0.8, min_samples=5)
db_labels = db.fit_predict(X)
n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise_db    = (db_labels == -1).sum()
if n_clusters_db > 1:
    mask = db_labels != -1
    db_sil = silhouette_score(X[mask], db_labels[mask])
else:
    db_sil = float("nan")

print("\nDBSCAN (eps=0.8) Results")
print("-" * 40)
print(f"  Clusters found: {n_clusters_db}")
print(f"  Noise points:   {n_noise_db}")
print(f"  Silhouette:     {db_sil:.4f}" if not np.isnan(db_sil) else "  Silhouette: N/A (< 2 clusters)")
for k in sorted(set(db_labels)):
    label = "Noise" if k == -1 else f"Cluster {k}"
    print(f"  {label}: {(db_labels == k).sum()} samples")

# --- PCA to 2D for visualization ---
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
ev = pca.explained_variance_ratio_
print(f"\nPCA explained variance: PC1={ev[0]:.2%}, PC2={ev[1]:.2%}, "
      f"total={ev.sum():.2%}")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors_km = ["steelblue", "darkorange", "green", "red"]

for ax, labels, title in zip(
        axes,
        [km_labels, db_labels],
        ["KMeans (k=4)", "DBSCAN (eps=0.8)"]):

    unique_labels = sorted(set(labels))
    palette = plt.cm.tab10.colors
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        color = "grey" if lbl == -1 else palette[i % len(palette)]
        name  = "Noise" if lbl == -1 else f"C{lbl}"
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=color, s=20, alpha=0.6, label=name)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8, markerscale=1.5)

plt.suptitle("Customer Segmentation (PCA 2D projection)", fontsize=12)
plt.tight_layout()
plt.savefig("ML/outputs/customer_segments.png", dpi=100)
plt.close()
print("Plot saved -> ML/outputs/customer_segments.png")
