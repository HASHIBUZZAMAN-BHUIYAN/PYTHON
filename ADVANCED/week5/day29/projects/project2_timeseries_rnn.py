"""
Day 29 Project 2: Time Series Prediction with RNN
==================================================
Generate a synthetic sine-wave time series (500 points).
Train an RNN to predict the next value given the past seq_len values.
Plot actual vs predicted.

Architecture: nn.RNN, hidden_size=16
~60 MB RAM, ~15s on CPU
"""

import torch
import torch.nn as nn
import numpy as np
import time

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("(matplotlib not found — skipping plots)")

# ── Generate Data ─────────────────────────────────────────────────────────────
np.random.seed(42)
N        = 500
t        = np.linspace(0, 4 * np.pi, N)
signal   = np.sin(t) + 0.1 * np.random.randn(N)
signal   = signal.astype(np.float32)

# Normalize to [-1, 1]
sig_min, sig_max = signal.min(), signal.max()
signal_norm = 2 * (signal - sig_min) / (sig_max - sig_min) - 1

SEQ_LEN = 20   # use 20 past values to predict the next one

def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_all, y_all = make_sequences(signal_norm, SEQ_LEN)

split    = int(0.8 * len(X_all))
X_train  = torch.tensor(X_all[:split]).unsqueeze(-1)   # [split, seq_len, 1]
y_train  = torch.tensor(y_all[:split]).unsqueeze(-1)   # [split, 1]
X_test   = torch.tensor(X_all[split:]).unsqueeze(-1)
y_test   = torch.tensor(y_all[split:]).unsqueeze(-1)

print(f"Train: {X_train.shape} → {y_train.shape}")
print(f"Test:  {X_test.shape}  → {y_test.shape}")

# ── Model ─────────────────────────────────────────────────────────────────────
class TimeSeriesRNN(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.fc(h_n.squeeze(0))

torch.manual_seed(42)
model     = TimeSeriesRNN(hidden_size=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params}")

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS     = 50
BATCH_SIZE = 32

print("\nTraining...")
start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(X_train))
    total_loss = 0.0
    n_batches  = 0

    for i in range(0, len(X_train), BATCH_SIZE):
        idx    = perm[i : i+BATCH_SIZE]
        xb, yb = X_train[idx], y_train[idx]
        pred   = model(xb)
        loss   = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_test), y_test).item()
        print(f"  Epoch {epoch:2d} | Train: {total_loss/n_batches:.4f} | Test: {test_loss:.4f}")

elapsed = time.time() - start
print(f"\nTraining complete in {elapsed:.1f}s")

# ── Evaluation ────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    y_pred = model(X_test).squeeze().numpy()
y_true = y_test.squeeze().numpy()

mse = float(np.mean((y_pred - y_true) ** 2))
mae = float(np.mean(np.abs(y_pred - y_true)))
print(f"\nTest MSE: {mse:.5f}")
print(f"Test MAE: {mae:.5f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
if HAS_PLOT:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle("RNN Time Series Prediction", fontsize=13)

    # Full signal
    axes[0].plot(signal_norm, color="steelblue", linewidth=0.8, label="Full signal")
    axes[0].axvline(split + SEQ_LEN, color="red", linestyle="--", label="Train/Test split")
    axes[0].set_title("Full Normalized Signal")
    axes[0].legend()

    # Test predictions
    test_indices = np.arange(len(y_true))
    axes[1].plot(test_indices, y_true, color="steelblue", label="Actual", linewidth=1)
    axes[1].plot(test_indices, y_pred, color="orange",    label="Predicted", linewidth=1, linestyle="--")
    axes[1].set_title(f"Test Set: Actual vs Predicted  (MSE={mse:.5f})")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("timeseries_rnn_prediction.png", dpi=100)
    print("\nPlot saved: timeseries_rnn_prediction.png")
    plt.show()
else:
    # text preview
    print("\nFirst 10 predictions vs actual (test set):")
    for i in range(10):
        print(f"  t={i}: actual={y_true[i]:.3f}  predicted={y_pred[i]:.3f}")
