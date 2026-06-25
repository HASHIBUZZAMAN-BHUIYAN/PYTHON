"""
What it does: Detects anomalies in synthetic sensor data (temperature, pressure)
              using Z-score, IsolationForest, and LocalOutlierFactor.
What it teaches: Classical anomaly detection approaches, comparing supervised-labeled
                 anomalies vs unsupervised detections, precision/recall evaluation.
Category: ANOMALY DETECTION
RAM estimate: < 100 MB
Time estimate: < 5 seconds
"""

import os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("ML/outputs", exist_ok=True)

np.random.seed(42)
n_normal   = 475
n_anomaly  = 25   # ~5%

# Normal sensor readings
temp_normal  = np.random.normal(70, 5,  n_normal)
pres_normal  = np.random.normal(100, 8, n_normal)

# Injected anomalies: high temperature or high pressure
temp_anom = np.random.uniform(110, 130, n_anomaly)
pres_anom = np.random.uniform(130, 160, n_anomaly)

temperature = np.concatenate([temp_normal, temp_anom])
pressure    = np.concatenate([pres_normal, pres_anom])
true_labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)]).astype(int)

X = np.column_stack([temperature, pressure])
print(f"Dataset: {len(X)} samples | {n_anomaly} true anomalies ({n_anomaly/len(X):.1%})")

# --- Method 1: Z-score (threshold = 3) ---
z_temp = np.abs(stats.zscore(temperature))
z_pres = np.abs(stats.zscore(pressure))
z_pred = ((z_temp > 3) | (z_pres > 3)).astype(int)

# --- Method 2: IsolationForest ---
iso = IsolationForest(contamination=0.05, random_state=42)
iso_raw  = iso.fit_predict(X)
iso_pred = (iso_raw == -1).astype(int)   # -1 = anomaly -> 1

# --- Method 3: LocalOutlierFactor (novelty=False for unsupervised) ---
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof_raw  = lof.fit_predict(X)
lof_pred = (lof_raw == -1).astype(int)

# --- Results table ---
methods = {
    "Z-score (thr=3)":      z_pred,
    "IsolationForest":      iso_pred,
    "LocalOutlierFactor":   lof_pred,
}

print("\nAnomaly Detection - Method Comparison")
print("-" * 68)
hdr = f"{'Method':<24} {'Detected':>8} {'Prec':>6} {'Rec':>6} {'F1':>6}"
print(hdr)
print("-" * 68)
for name, pred in methods.items():
    n_det = pred.sum()
    prec  = precision_score(true_labels, pred, zero_division=0)
    rec   = recall_score(true_labels, pred, zero_division=0)
    f1    = f1_score(true_labels, pred, zero_division=0)
    print(f"{name:<24} {n_det:>8} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")
print("-" * 68)

# --- 2D scatter plot ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = list(methods.keys())

for ax, (name, pred) in zip(axes, methods.items()):
    normal_mask  = pred == 0
    anomaly_mask = pred == 1
    ax.scatter(temperature[normal_mask],  pressure[normal_mask],
               color="steelblue", s=15, alpha=0.5, label="Normal")
    ax.scatter(temperature[anomaly_mask], pressure[anomaly_mask],
               color="red", s=40, alpha=0.8, marker="x", label="Anomaly", linewidths=1.5)
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Pressure")
    ax.set_title(name)
    ax.legend(fontsize=8)

plt.suptitle("Anomaly Detection - Synthetic Sensor Data", fontsize=12)
plt.tight_layout()
plt.savefig("ML/outputs/anomaly_detection.png", dpi=100)
plt.close()
print("Plot saved -> ML/outputs/anomaly_detection.png")
