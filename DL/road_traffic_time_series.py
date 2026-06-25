"""
road_traffic_time_series.py
===========================
What it does:
    Generates 2 years of synthetic hourly traffic volume data, then trains a
    small PyTorch LSTM to forecast the next hour given the previous 24 hours.

What it teaches:
    - Time-series feature engineering (sliding windows)
    - LSTM architecture for sequence prediction
    - Evaluation with MAE and RMSE
    - Plotting actual vs predicted values

RAM estimate : ~30 MB
Time estimate: ~30 seconds on CPU (Ryzen 7)
Real vs simulated: ALL data is synthetically generated in-code.
    No files are downloaded. Patterns mimic real traffic cycles
    (rush hours, weekday/weekend) but are fully artificial.
"""

import os
import math
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

# ── 1. Synthetic traffic data generation ──────────────────────────────────────
def generate_traffic(seed=42):
    rng = np.random.default_rng(seed)
    hours = 24 * 365 * 2          # 2 years of hourly readings
    t = np.arange(hours)

    hour_of_day  = t % 24
    day_of_week  = (t // 24) % 7

    # Daily pattern: two peaks (morning ~8h, evening ~17h)
    morning_peak = np.exp(-0.5 * ((hour_of_day - 8)  / 1.5) ** 2)
    evening_peak = np.exp(-0.5 * ((hour_of_day - 17) / 1.5) ** 2)
    daily_pattern = 400 * morning_peak + 350 * evening_peak

    # Weekly pattern: weekdays higher than weekends
    is_weekday = (day_of_week < 5).astype(float)
    weekly_pattern = 200 * is_weekday + 50 * (1 - is_weekday)

    # Base traffic + patterns + noise
    base = 300.0
    noise = rng.normal(0, 30, hours)
    traffic = base + daily_pattern + weekly_pattern + noise
    traffic = np.clip(traffic, 0, None).astype(np.float32)
    return traffic

traffic = generate_traffic()
print(f"[OK] Traffic data shape: {traffic.shape}  min={traffic.min():.1f}  max={traffic.max():.1f}")

# ── 2. Sliding window dataset ──────────────────────────────────────────────────
SEQ_LEN = 24   # use 24 hours to predict next hour

def make_sequences(series, seq_len):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i : i + seq_len])
        y.append(series[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X, y = make_sequences(traffic, SEQ_LEN)

# Normalise using training stats (avoid leakage)
split = int(len(X) * 0.8)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

mean_ = X_tr.mean()
std_  = X_tr.std() + 1e-8

X_tr = (X_tr - mean_) / std_
X_te = (X_te - mean_) / std_
y_tr = (y_tr - mean_) / std_
y_te = (y_te - mean_) / std_

# PyTorch tensors  ->  (N, seq_len, 1)
X_tr_t = torch.from_numpy(X_tr).unsqueeze(-1)
y_tr_t = torch.from_numpy(y_tr).unsqueeze(-1)
X_te_t = torch.from_numpy(X_te).unsqueeze(-1)
y_te_t = torch.from_numpy(y_te).unsqueeze(-1)

train_loader = DataLoader(
    TensorDataset(X_tr_t, y_tr_t), batch_size=256, shuffle=True
)

print(f"[OK] Train: {X_tr_t.shape}  Test: {X_te_t.shape}")

# ── 3. LSTM model ──────────────────────────────────────────────────────────────
class TrafficLSTM(nn.Module):
    def __init__(self, hidden=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden,
            num_layers=num_layers, batch_first=True, dropout=0.1
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # last time-step

model = TrafficLSTM(hidden=32, num_layers=2)
total_params = sum(p.numel() for p in model.parameters())
print(f"[OK] LSTM params: {total_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ── 4. Training loop ───────────────────────────────────────────────────────────
EPOCHS = 10
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(xb)
    avg_loss = running_loss / len(X_tr_t)
    if epoch == 1 or epoch % 2 == 0:
        print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}")

elapsed = time.time() - t0
print(f"[OK] Training done in {elapsed:.1f}s")

# ── 5. Evaluation ─────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    preds_norm = model(X_te_t).squeeze().numpy()
    actuals_norm = y_te_t.squeeze().numpy()

# Denormalise
preds   = preds_norm   * std_ + mean_
actuals = actuals_norm * std_ + mean_

mae  = np.mean(np.abs(preds - actuals))
rmse = np.sqrt(np.mean((preds - actuals) ** 2))
print(f"\n[OK] Test MAE : {mae:.2f} vehicles/hour")
print(f"[OK] Test RMSE: {rmse:.2f} vehicles/hour")

# ── 6. Plot last 2 weeks of test data ─────────────────────────────────────────
if matplotlib_available:
    two_weeks = 24 * 14
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(actuals[-two_weeks:], label="Actual",    color="steelblue",  lw=1.2)
    ax.plot(preds[-two_weeks:],   label="Predicted", color="orangered", lw=1.0, alpha=0.8)
    ax.set_title("Traffic Forecast - Last 2 Weeks of Test Set (LSTM)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Traffic Volume (vehicles/hour)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("DL/outputs/traffic_forecast.png", dpi=100)
    plt.close()
    print("[OK] Plot saved to DL/outputs/traffic_forecast.png")
else:
    print("[X] matplotlib not available, skipping plot")

print("\n[DONE] road_traffic_time_series.py complete")
