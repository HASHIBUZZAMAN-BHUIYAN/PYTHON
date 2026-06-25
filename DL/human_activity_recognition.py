"""
human_activity_recognition.py
==============================
What it does:
    Generates synthetic 3-axis accelerometer windows for three activities
    (walking, running, sitting) and trains a 1D-CNN classifier.

What it teaches:
    - Synthetic sensor data generation with realistic physical characteristics
    - 1D-CNN for time-series classification
    - Confusion matrix visualisation
    - Per-class accuracy reporting

RAM estimate : ~50 MB
Time estimate: ~20 seconds on CPU (Ryzen 7)
Real vs simulated: ALL data is synthetically generated in-code.
    Amplitude and frequency values mimic realistic accelerometer readings
    but no real sensor recordings are used.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

matplotlib_available = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib_available = False

os.makedirs("DL/outputs", exist_ok=True)

# ── 1. Synthetic accelerometer data ───────────────────────────────────────────
WINDOW  = 128   # samples per window
N_CLASS = 500   # windows per class
FS      = 50.0  # sampling rate Hz
rng     = np.random.default_rng(42)

def make_activity(activity, n_windows, window_len, rng):
    t = np.linspace(0, window_len / FS, window_len)
    data = []
    for _ in range(n_windows):
        if activity == "walking":
            freq  = 2.0 + rng.uniform(-0.2, 0.2)
            amp   = 0.75 + rng.uniform(-0.1, 0.1)
            x = amp * np.sin(2 * np.pi * freq * t) + rng.normal(0, 0.05, window_len)
            y = amp * 0.6 * np.sin(2 * np.pi * freq * t + 0.5) + rng.normal(0, 0.05, window_len)
            z = 1.0 + rng.normal(0, 0.08, window_len)
        elif activity == "running":
            freq  = 3.0 + rng.uniform(-0.3, 0.3)
            amp   = 2.0 + rng.uniform(-0.2, 0.2)
            x = amp * np.sin(2 * np.pi * freq * t) + rng.normal(0, 0.15, window_len)
            y = amp * 0.8 * np.sin(2 * np.pi * freq * t + 0.4) + rng.normal(0, 0.15, window_len)
            z = 1.0 + amp * 0.3 * np.sin(2 * np.pi * freq * t) + rng.normal(0, 0.1, window_len)
        else:  # sitting
            x = rng.normal(0,   0.05, window_len)
            y = rng.normal(0,   0.05, window_len)
            z = rng.normal(1.0, 0.05, window_len)
        data.append(np.stack([x, y, z], axis=0).astype(np.float32))
    return np.array(data)   # (n_windows, 3, window_len)

walking = make_activity("walking", N_CLASS, WINDOW, rng)
running = make_activity("running", N_CLASS, WINDOW, rng)
sitting = make_activity("sitting", N_CLASS, WINDOW, rng)

X = np.concatenate([walking, running, sitting], axis=0)
y = np.array([0]*N_CLASS + [1]*N_CLASS + [2]*N_CLASS, dtype=np.int64)
CLASS_NAMES = ["walking", "running", "sitting"]
print(f"[OK] Dataset shape: X={X.shape}  labels unique={np.unique(y)}")

# Shuffle
idx = rng.permutation(len(X))
X, y = X[idx], y[idx]

X_t = torch.from_numpy(X)
y_t = torch.from_numpy(y)

dataset = TensorDataset(X_t, y_t)
n_train = int(0.8 * len(dataset))
n_test  = len(dataset) - n_train
train_ds, test_ds = random_split(dataset, [n_train, n_test],
                                 generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

# ── 2. 1D-CNN model ───────────────────────────────────────────────────────────
class ActivityCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(3,  16, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(16)
        self.bn2   = nn.BatchNorm1d(32)
        self.pool  = nn.AdaptiveAvgPool1d(1)   # global average pool
        self.fc    = nn.Linear(32, n_classes)
        self.drop  = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return self.fc(x)

model     = ActivityCNN()
total_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
print(f"[OK] CNN params: {total_params:,}")

# ── 3. Training ────────────────────────────────────────────────────────────────
EPOCHS = 10
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    correct, total, running_loss = 0, 0, 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(xb)
        correct += (out.argmax(1) == yb).sum().item()
        total   += len(yb)
    if epoch == 1 or epoch % 2 == 0:
        print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={running_loss/total:.4f}  "
              f"train_acc={correct/total:.3f}")

elapsed = time.time() - t0
print(f"[OK] Training done in {elapsed:.1f}s")

# ── 4. Evaluation ─────────────────────────────────────────────────────────────
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb).argmax(1)
        all_preds.extend(preds.numpy())
        all_labels.extend(yb.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
overall_acc = (all_preds == all_labels).mean()
print(f"\n[OK] Overall test accuracy: {overall_acc:.3f}")

# Per-class accuracy
for i, name in enumerate(CLASS_NAMES):
    mask = all_labels == i
    acc_i = (all_preds[mask] == all_labels[mask]).mean() if mask.sum() > 0 else 0.0
    print(f"  {name:8s}: {acc_i:.3f}  (n={mask.sum()})")

# Confusion matrix
n_classes = len(CLASS_NAMES)
conf_mat = np.zeros((n_classes, n_classes), dtype=int)
for true, pred in zip(all_labels, all_preds):
    conf_mat[true, pred] += 1

print("\nConfusion matrix (rows=actual, cols=predicted):")
print("        " + "  ".join(f"{c:8s}" for c in CLASS_NAMES))
for i, row in enumerate(conf_mat):
    print(f"  {CLASS_NAMES[i]:8s}" + "  ".join(f"{v:8d}" for v in row))

# ── 5. Plot confusion matrix ───────────────────────────────────────────────────
if matplotlib_available:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(conf_mat, cmap="Blues")
    ax.set_xticks(range(n_classes)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(n_classes)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Activity Recognition Confusion Matrix\n(Overall Acc={overall_acc:.3f})")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(conf_mat[i, j]), ha="center", va="center",
                    color="white" if conf_mat[i, j] > conf_mat.max() * 0.5 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("DL/outputs/activity_recognition.png", dpi=100)
    plt.close()
    print("[OK] Confusion matrix saved to DL/outputs/activity_recognition.png")

print("\n[DONE] human_activity_recognition.py complete")
