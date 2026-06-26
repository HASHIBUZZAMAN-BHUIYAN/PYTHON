"""
anomaly_detection_realworld.py
================================
What it does:
    Simulates industrial machine sensor data (temperature + vibration),
    injects anomalies at ~5% of timesteps, trains a 1D-convolutional
    autoencoder ONLY on normal data, then flags anomalies via reconstruction
    error thresholding.

What it teaches:
    - Unsupervised anomaly detection with autoencoders
    - 1D-conv encoder-decoder architecture
    - Threshold selection (95th percentile of training reconstruction errors)
    - Precision / recall evaluation

RAM estimate : ~30 MB
Time estimate: ~20 seconds on CPU (Ryzen 7)
Real vs simulated: ALL sensor readings are synthetically generated in-code.
    Amplitude/frequency values mimic realistic industrial sensors but no
    real machine data is downloaded.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib_available = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib_available = False

os.makedirs("DL/outputs", exist_ok=True)
rng = np.random.default_rng(42)

# ── 1. Generate synthetic sensor data ─────────────────────────────────────────
N_SAMPLES  = 2000
ANOMALY_RATE = 0.05
t = np.linspace(0, 200, N_SAMPLES)

# Temperature: base 70 deg C + slow sine drift + small noise
temp_normal = 70.0 + 5.0 * np.sin(2 * np.pi * t / 50) + rng.normal(0, 0.5, N_SAMPLES)

# Vibration: base 0.3g + 2Hz oscillation + small noise
vib_normal  = 0.3 + 0.1 * np.sin(2 * np.pi * 2 * t) + rng.normal(0, 0.02, N_SAMPLES)

anomaly_mask = np.zeros(N_SAMPLES, dtype=bool)
n_anomalies  = int(N_SAMPLES * ANOMALY_RATE)
anomaly_idx  = rng.choice(N_SAMPLES, size=n_anomalies, replace=False)
anomaly_mask[anomaly_idx] = True

temp = temp_normal.copy()
vib  = vib_normal.copy()
temp[anomaly_mask] += rng.uniform(20, 40, n_anomalies)     # +20..+40 deg
vib[anomaly_mask]  += rng.uniform(1.5, 3.0, n_anomalies)   # +1.5..+3g

signal = np.stack([temp, vib], axis=1).astype(np.float32)

# Normalise using only normal-data statistics
normal_mean = signal[~anomaly_mask].mean(axis=0)
normal_std  = signal[~anomaly_mask].std(axis=0) + 1e-8
signal_norm = (signal - normal_mean) / normal_std

print(f"[OK] Signal shape: {signal.shape}  "
      f"anomalies: {anomaly_mask.sum()} ({anomaly_mask.mean():.1%})")

# ── 2. Sliding window dataset ──────────────────────────────────────────────────
WINDOW = 32   # 32-step windows

def make_windows(data, mask, window_len):
    """Return windows of shape (N, channels, window_len) and per-window labels."""
    X_list, label_list = [], []
    for i in range(len(data) - window_len):
        win = data[i : i + window_len]          # (window_len, 2)
        X_list.append(win.T)                     # (2, window_len)
        # label = 1 if ANY timestep in window is anomalous
        label_list.append(int(mask[i : i + window_len].any()))
    return np.array(X_list, dtype=np.float32), np.array(label_list, dtype=np.int32)

X_all, labels_all = make_windows(signal_norm, anomaly_mask, WINDOW)
print(f"[OK] Windows: {X_all.shape}  anomalous windows: {labels_all.sum()}")

normal_windows = X_all[labels_all == 0]
anom_windows   = X_all[labels_all == 1]
print(f"[OK] Normal windows: {len(normal_windows)}  Anomaly windows: {len(anom_windows)}")

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(normal_windows)),
    batch_size=64, shuffle=True
)

# ── 3. 1D Convolutional Autoencoder ───────────────────────────────────────────
class ConvAutoencoder(nn.Module):
    """
    Encoder: 2 -> 8 -> 16 channels (halving length with stride-2 convs)
    Decoder: 16 -> 8 -> 2 channels (doubling length with ConvTranspose1d)
    Input/output shape: (B, 2, 32)
    """
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv1d(2,  8,  kernel_size=3, stride=2, padding=1)   # ->16
        self.enc2 = nn.Conv1d(8,  16, kernel_size=3, stride=2, padding=1)   # ->8
        self.bn_e1 = nn.BatchNorm1d(8)
        self.bn_e2 = nn.BatchNorm1d(16)
        self.dec1 = nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1)  # ->16
        self.dec2 = nn.ConvTranspose1d(8,  2, kernel_size=4, stride=2, padding=1)  # ->32
        self.bn_d1 = nn.BatchNorm1d(8)

    def forward(self, x):
        z = torch.relu(self.bn_e1(self.enc1(x)))
        z = torch.relu(self.bn_e2(self.enc2(z)))
        z = torch.relu(self.bn_d1(self.dec1(z)))
        return self.dec2(z)   # (B, 2, 32) reconstruction

model = ConvAutoencoder()
total_params = sum(p.numel() for p in model.parameters())
print(f"[OK] Autoencoder params: {total_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ── 4. Training ────────────────────────────────────────────────────────────────
EPOCHS = 15
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, total = 0.0, 0
    for (xb,) in train_loader:
        optimizer.zero_grad()
        recon = model(xb)
        loss  = criterion(recon, xb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(xb)
        total += len(xb)
    if epoch == 1 or epoch % 3 == 0:
        print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={running_loss/total:.5f}")

elapsed = time.time() - t0
print(f"[OK] Training done in {elapsed:.1f}s")

# ── 5. Compute reconstruction errors ──────────────────────────────────────────
model.eval()

def recon_error(windows_np):
    """MSE per window (mean over channels and time)."""
    with torch.no_grad():
        t_in   = torch.from_numpy(windows_np)
        t_out  = model(t_in)
        errors = ((t_in - t_out) ** 2).mean(dim=(1, 2)).numpy()
    return errors

train_errors = recon_error(normal_windows)
threshold    = np.percentile(train_errors, 95)   # 95th pct sets the anomaly cutoff
print(f"\n[OK] Threshold (95th pct of training errors): {threshold:.5f}")

all_errors = recon_error(X_all)
predictions = (all_errors > threshold).astype(int)
true_labels = labels_all.astype(int)

tp = int(((predictions == 1) & (true_labels == 1)).sum())
fp = int(((predictions == 1) & (true_labels == 0)).sum())
fn = int(((predictions == 0) & (true_labels == 1)).sum())
precision = tp / (tp + fp + 1e-8)
recall    = tp / (tp + fn + 1e-8)
f1        = 2 * precision * recall / (precision + recall + 1e-8)

print(f"[OK] Precision : {precision:.3f}")
print(f"[OK] Recall    : {recall:.3f}")
print(f"[OK] F1 Score  : {f1:.3f}")
print(f"     TP={tp}  FP={fp}  FN={fn}")

# ── 6. Plots ───────────────────────────────────────────────────────────────────
if matplotlib_available:
    all_recon = []
    model.eval()
    with torch.no_grad():
        t_in  = torch.from_numpy(X_all)
        t_out = model(t_in)
        all_recon = t_out[:, 0, -1].numpy()   # last timestep, channel 0 (temp)

    # Align: each window's reconstruction refers to timestep i+WINDOW-1
    plot_len   = min(500, len(X_all))
    xs         = np.arange(plot_len)
    actual_temp_norm = signal_norm[WINDOW - 1: WINDOW - 1 + plot_len, 0]
    recon_temp_norm  = all_recon[:plot_len]
    errors_plot      = all_errors[:plot_len]
    anom_flags       = (predictions[:plot_len] == 1)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(xs, actual_temp_norm, color="steelblue", lw=0.8, label="Actual temp (norm)")
    axes[0].set_ylabel("Temp (norm)")
    axes[0].set_title("Industrial Sensor Anomaly Detection (Autoencoder)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, recon_temp_norm, color="darkorange", lw=0.8, label="Reconstructed temp")
    axes[1].set_ylabel("Recon (norm)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(xs, errors_plot, color="purple", lw=0.8, label="Recon Error")
    axes[2].axhline(threshold, color="red", ls="--", lw=1.0, label=f"Threshold={threshold:.4f}")
    axes[2].set_ylabel("MSE Error")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    axes[3].fill_between(xs, 0, anom_flags.astype(float),
                         color="red", alpha=0.6, label="Predicted anomaly")
    true_anom_plot = true_labels[:plot_len].astype(float)
    axes[3].fill_between(xs, 0, true_anom_plot,
                         color="blue", alpha=0.3, label="True anomaly")
    axes[3].set_ylabel("Anomaly Flag")
    axes[3].set_xlabel("Window index")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("DL/outputs/anomaly_detection.png", dpi=100)
    plt.close()
    print("[OK] Plot saved to DL/outputs/anomaly_detection.png")

print("\n[DONE] anomaly_detection_realworld.py complete")
