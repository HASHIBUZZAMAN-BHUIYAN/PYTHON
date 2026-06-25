"""
Day 30 Project 2: Time Series Forecasting with LSTM
====================================================
Synthetic stock-like data: trend + seasonality + noise (500 points).
LSTM: hidden=32, seq_len=10.
Predict next 10 steps (multi-step forecast).
Plot forecast vs actual for the test period.

~80 MB RAM, ~15s on CPU
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
    print("(matplotlib not available — showing text output only)")

torch.manual_seed(42)
np.random.seed(42)

# ── Generate Synthetic Data ───────────────────────────────────────────────────
N      = 500
t      = np.linspace(0, 4 * np.pi, N).astype(np.float32)
trend  = 0.02 * np.arange(N, dtype=np.float32)
season = np.sin(t).astype(np.float32)
noise  = 0.15 * np.random.randn(N).astype(np.float32)
signal = trend + season + noise

# Normalize to [0, 1]
sig_min, sig_max = signal.min(), signal.max()
signal_norm = (signal - sig_min) / (sig_max - sig_min)

print(f"Signal: {N} points, range [{signal_norm.min():.3f}, {signal_norm.max():.3f}]")

# ── Create Sliding-Window Sequences ──────────────────────────────────────────
SEQ_LEN    = 10
PRED_STEPS = 10   # predict next 10 steps

def make_multi_step(data, seq_len, pred_steps):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_steps + 1):
        X.append(data[i : i+seq_len])
        y.append(data[i+seq_len : i+seq_len+pred_steps])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_all, y_all = make_multi_step(signal_norm, SEQ_LEN, PRED_STEPS)
print(f"Sequences: X={X_all.shape}, y={y_all.shape}")

split   = int(0.8 * len(X_all))
X_train = torch.tensor(X_all[:split]).unsqueeze(-1)   # [split, 10, 1]
y_train = torch.tensor(y_all[:split])                  # [split, 10]
X_test  = torch.tensor(X_all[split:]).unsqueeze(-1)
y_test  = torch.tensor(y_all[split:])
print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

# ── Model ─────────────────────────────────────────────────────────────────────
class ForecastLSTM(nn.Module):
    def __init__(self, hidden_size=32, pred_steps=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, pred_steps)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)          # h_n: [1, batch, hidden]
        return self.fc(h_n.squeeze(0))       # [batch, pred_steps]


model     = ForecastLSTM(hidden_size=32, pred_steps=PRED_STEPS)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params}")

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS     = 60
BATCH_SIZE = 32

print("Training LSTM forecaster...")
start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    perm       = torch.randperm(len(X_train))
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

    if epoch % 15 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_test), y_test).item()
        print(f"  Epoch {epoch:2d} | Train: {total_loss/n_batches:.5f} | Test: {test_loss:.5f}")

elapsed = time.time() - start
print(f"\nTraining complete in {elapsed:.1f}s")

# ── Evaluation ────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    y_pred = model(X_test).numpy()   # [n_test, 10]

y_true = y_test.numpy()              # [n_test, 10]

mse = float(np.mean((y_pred - y_true) ** 2))
mae = float(np.mean(np.abs(y_pred - y_true)))
print(f"\nTest MSE: {mse:.6f}")
print(f"Test MAE: {mae:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
if HAS_PLOT:
    # Show the last test sample: input window + actual future + predicted future
    sample_idx    = -1   # last test sample
    input_window  = X_test[sample_idx].squeeze().numpy()    # [10]
    actual_future = y_true[sample_idx]                       # [10]
    pred_future   = y_pred[sample_idx]                       # [10]

    x_input = np.arange(SEQ_LEN)
    x_future = np.arange(SEQ_LEN, SEQ_LEN + PRED_STEPS)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("LSTM Time Series Forecast", fontsize=13)

    # Left: last test sample detail
    axes[0].plot(x_input,  input_window,  color="steelblue", marker="o", ms=4, label="Input window")
    axes[0].plot(x_future, actual_future, color="green",  marker="o", ms=4, label="Actual future")
    axes[0].plot(x_future, pred_future,   color="orange", marker="x", ms=5, linestyle="--", label="Predicted")
    axes[0].axvline(SEQ_LEN - 0.5, color="gray", linestyle=":")
    axes[0].set_title("Last Test Sample: 10-Step Forecast")
    axes[0].legend()

    # Right: full test set — first predicted step vs actual
    axes[1].plot(y_true[:, 0], color="steelblue", label="Actual (step+1)", linewidth=1)
    axes[1].plot(y_pred[:, 0], color="orange",    label="Predicted (step+1)", linewidth=1, linestyle="--")
    axes[1].set_title(f"Test Set — 1-Step-Ahead (MSE={mse:.5f})")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("timeseries_lstm_forecast.png", dpi=100)
    print("\nPlot saved: timeseries_lstm_forecast.png")
    plt.show()
else:
    print("\nSample forecast (last test window):")
    actual = y_true[-1]
    pred   = y_pred[-1]
    for step in range(PRED_STEPS):
        print(f"  Step {step+1:2d}: actual={actual[step]:.3f}  predicted={pred[step]:.3f}")
