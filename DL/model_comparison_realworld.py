"""
model_comparison_realworld.py
==============================
What it does:
    Recreates the synthetic hourly traffic dataset from road_traffic_time_series.py
    (same seed), then benchmarks four forecasting approaches:
      1. Naive (persistence) baseline
      2. Moving-average baseline (window=24)
      3. Linear regression (sklearn, lag features)
      4. LSTM (same tiny architecture as file 1)

    Prints a comparison table and saves a bar chart.

What it teaches:
    - Why baselines always come first before committing to deep learning
    - The cost/benefit of model complexity vs. accuracy gain
    - How simple lag features can be competitive for periodic data
    - When LSTMs add value vs when baselines suffice

RAM estimate : ~50 MB
Time estimate: ~35 seconds on CPU (Ryzen 7)
Real vs simulated: ALL data is synthetically generated in-code using the
    same seed as road_traffic_time_series.py for reproducibility.
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

from sklearn.linear_model import Ridge

os.makedirs("DL/outputs", exist_ok=True)

# ── 1. Traffic data (identical generation to file 1) ──────────────────────────
def generate_traffic(seed=42):
    rng = np.random.default_rng(seed)
    hours = 24 * 365 * 2
    t = np.arange(hours)
    hour_of_day = t % 24
    day_of_week = (t // 24) % 7
    morning_peak = np.exp(-0.5 * ((hour_of_day - 8)  / 1.5) ** 2)
    evening_peak = np.exp(-0.5 * ((hour_of_day - 17) / 1.5) ** 2)
    daily_pattern   = 400 * morning_peak + 350 * evening_peak
    is_weekday      = (day_of_week < 5).astype(float)
    weekly_pattern  = 200 * is_weekday + 50 * (1 - is_weekday)
    base = 300.0
    noise = rng.normal(0, 30, hours)
    traffic = base + daily_pattern + weekly_pattern + noise
    return np.clip(traffic, 0, None).astype(np.float32)

traffic = generate_traffic(seed=42)
SEQ_LEN = 24
N = len(traffic)
split = int(N * 0.8)

y_true = traffic[SEQ_LEN:]
y_true_tr = y_true[:split - SEQ_LEN]
y_true_te = y_true[split - SEQ_LEN:]

results = {}

# ── 2. Method 1: Naive persistence (predict last known value) ─────────────────
t0 = time.time()
naive_preds_te = traffic[split - 1 : split - 1 + len(y_true_te)]   # lag-1
naive_time = time.time() - t0
mae_naive  = np.mean(np.abs(naive_preds_te - y_true_te))
rmse_naive = np.sqrt(np.mean((naive_preds_te - y_true_te) ** 2))
results["Naive (persist)"] = dict(mae=mae_naive, rmse=rmse_naive, train_time=naive_time)
print(f"[OK] Naive  MAE={mae_naive:.2f}  RMSE={rmse_naive:.2f}  time={naive_time:.3f}s")

# ── 3. Method 2: Moving average (window=24) ───────────────────────────────────
t0 = time.time()
cumsum = np.cumsum(np.concatenate([[0], traffic]))
roll   = (cumsum[SEQ_LEN:] - cumsum[:-SEQ_LEN]) / SEQ_LEN   # 24-step MA
ma_preds_te = roll[split - SEQ_LEN : split - SEQ_LEN + len(y_true_te)]
ma_time = time.time() - t0
mae_ma  = np.mean(np.abs(ma_preds_te - y_true_te))
rmse_ma = np.sqrt(np.mean((ma_preds_te - y_true_te) ** 2))
results["Moving Avg (24h)"] = dict(mae=mae_ma, rmse=rmse_ma, train_time=ma_time)
print(f"[OK] MovAvg MAE={mae_ma:.2f}  RMSE={rmse_ma:.2f}  time={ma_time:.3f}s")

# ── 4. Method 3: Linear Regression with lag features ─────────────────────────
def build_features(series, seq_len):
    """Features: lag_1, lag_24, hour_of_day, day_of_week."""
    X_list, y_list = [], []
    for i in range(seq_len, len(series)):
        lag1  = series[i - 1]
        lag24 = series[i - seq_len]
        hod   = i % 24
        dow   = (i // 24) % 7
        X_list.append([lag1, lag24, hod, dow])
        y_list.append(series[i])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

X_feat, y_feat = build_features(traffic, SEQ_LEN)
X_feat_tr, y_feat_tr = X_feat[:split - SEQ_LEN], y_feat[:split - SEQ_LEN]
X_feat_te, y_feat_te = X_feat[split - SEQ_LEN:], y_feat[split - SEQ_LEN:]

t0 = time.time()
lr_model = Ridge(alpha=1.0)
lr_model.fit(X_feat_tr, y_feat_tr)
lr_train_time = time.time() - t0

lr_preds_te = lr_model.predict(X_feat_te)
mae_lr  = np.mean(np.abs(lr_preds_te - y_feat_te))
rmse_lr = np.sqrt(np.mean((lr_preds_te - y_feat_te) ** 2))
results["Linear Reg (lags)"] = dict(mae=mae_lr, rmse=rmse_lr, train_time=lr_train_time)
print(f"[OK] LinReg MAE={mae_lr:.2f}  RMSE={rmse_lr:.2f}  train_time={lr_train_time:.3f}s")

# ── 5. Method 4: LSTM (same architecture as file 1) ───────────────────────────
def make_sequences(series, seq_len):
    X_s, y_s = [], []
    for i in range(len(series) - seq_len):
        X_s.append(series[i : i + seq_len])
        y_s.append(series[i + seq_len])
    return np.array(X_s, dtype=np.float32), np.array(y_s, dtype=np.float32)

X_seq, y_seq = make_sequences(traffic, SEQ_LEN)
X_seq_tr, y_seq_tr = X_seq[:split - SEQ_LEN], y_seq[:split - SEQ_LEN]
X_seq_te, y_seq_te = X_seq[split - SEQ_LEN:], y_seq[split - SEQ_LEN:]

mean_ = X_seq_tr.mean(); std_ = X_seq_tr.std() + 1e-8
X_seq_tr_n = (X_seq_tr - mean_) / std_
X_seq_te_n = (X_seq_te - mean_) / std_
y_seq_tr_n = (y_seq_tr - mean_) / std_

X_tr_t = torch.from_numpy(X_seq_tr_n).unsqueeze(-1)
y_tr_t = torch.from_numpy(y_seq_tr_n).unsqueeze(-1)
X_te_t = torch.from_numpy(X_seq_te_n).unsqueeze(-1)

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=256, shuffle=True)

class TrafficLSTM(nn.Module):
    def __init__(self, hidden=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=num_layers,
                            batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

lstm_model = TrafficLSTM()
optimizer  = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
criterion  = nn.MSELoss()

t0 = time.time()
EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    lstm_model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(lstm_model(xb), yb)
        loss.backward()
        optimizer.step()
    if epoch == 1 or epoch % 3 == 0:
        print(f"  LSTM epoch {epoch:2d}/{EPOCHS}")

lstm_train_time = time.time() - t0

lstm_model.eval()
with torch.no_grad():
    lstm_preds_n = lstm_model(X_te_t).squeeze().numpy()
lstm_preds = lstm_preds_n * std_ + mean_

mae_lstm  = np.mean(np.abs(lstm_preds - y_seq_te))
rmse_lstm = np.sqrt(np.mean((lstm_preds - y_seq_te) ** 2))
results["LSTM (2-layer)"] = dict(mae=mae_lstm, rmse=rmse_lstm, train_time=lstm_train_time)
print(f"[OK] LSTM   MAE={mae_lstm:.2f}  RMSE={rmse_lstm:.2f}  train_time={lstm_train_time:.1f}s")

# ── 6. Comparison table ────────────────────────────────────────────────────────
print("\n" + "-" * 62)
print(f"{'Method':<22} {'MAE':>8} {'RMSE':>8} {'Train (s)':>10}")
print("-" * 62)
for name, d in results.items():
    print(f"{name:<22} {d['mae']:>8.2f} {d['rmse']:>8.2f} {d['train_time']:>10.3f}")
print("-" * 62)

print("""
Discussion
----------
* Naive (persistence): Shockingly competitive for hourly traffic because
  traffic volume is highly autocorrelated over 1 step. RMSE is a rough
  upper bound; anything worse is worse than doing nothing.

* Moving average (24h): Captures the daily cycle well. Better RMSE than
  naive because it smooths noise, but lags behind sudden changes.

* Linear regression (lags): Adding hour-of-day + day-of-week as features
  gives a big boost for free. Often matches or beats LSTM on such periodic
  data with far less compute. Interpretable and fast to retrain.

* LSTM: Adds value when the signal has NON-LINEAR, LONG-RANGE dependencies
  that feature engineering cannot capture (e.g. events, cascades). On
  clean synthetic sinusoidal data it has diminishing returns vs. LinReg.
  Real-world benefit: irregular events, multi-step ahead forecasts, or
  when many covariates interact in complex ways.

Rule of thumb: Always beat a strong baseline (LinReg + hand features)
before deploying a neural network. The infrastructure cost is rarely
justified for a 2% RMSE reduction.
""")

# ── 7. Bar chart ───────────────────────────────────────────────────────────────
if matplotlib_available:
    methods = list(results.keys())
    mae_vals  = [results[m]["mae"]  for m in methods]
    rmse_vals = [results[m]["rmse"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, mae_vals,  width, label="MAE",  color="steelblue")
    bars2 = ax.bar(x + width/2, rmse_vals, width, label="RMSE", color="salmon")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Error (vehicles/hour)")
    ax.set_title("Traffic Forecasting: Baseline vs DL Comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig("DL/outputs/model_comparison.png", dpi=100)
    plt.close()
    print("[OK] Bar chart saved to DL/outputs/model_comparison.png")

print("\n[DONE] model_comparison_realworld.py complete")
