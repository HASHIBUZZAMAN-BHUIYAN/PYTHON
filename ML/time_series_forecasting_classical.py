"""
What it does: Forecasts synthetic daily sales data using four classical methods:
              Naive, Moving Average, Exponential Smoothing, and Decomposition+LinTrend.
What it teaches: Classical time-series forecasting without neural networks,
                 train/test split for time series, MAE/RMSE evaluation, seasonality.
Category: TIME-SERIES FORECASTING (CLASSICAL)
RAM estimate: < 100 MB
Time estimate: < 5 seconds
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("ML/outputs", exist_ok=True)

np.random.seed(42)

# --- Generate 2 years of daily synthetic sales data ---
n_days = 365 * 2   # 730 days
t      = np.arange(n_days)

trend     = 0.05 * t                           # gentle upward trend
weekly    = 15 * np.sin(2 * np.pi * t / 7)    # weekly seasonality
annual    = 40 * np.sin(2 * np.pi * t / 365)  # annual seasonality
noise     = np.random.normal(0, 8, n_days)

sales = 200 + trend + weekly + annual + noise
sales = np.clip(sales, 50, None)               # no negative sales

print(f"Sales series: {n_days} days  |  mean={sales.mean():.1f}  std={sales.std():.1f}")

# --- Train / Test split (80% / 20%) ---
split     = int(n_days * 0.8)
y_train   = sales[:split]
y_test    = sales[split:]
n_test    = len(y_test)
print(f"Train: {split} days  |  Test: {n_test} days")


# ------- Forecasting methods -------

# 1. Naive: last known value repeated
naive_forecast = np.full(n_test, y_train[-1])

# 2. Moving Average (window=7)
window = 7
ma_forecast = np.full(n_test, np.mean(y_train[-window:]))
# Recursive one-step-ahead
buffer = list(y_train[-window:])
ma_preds = []
for i in range(n_test):
    pred = np.mean(buffer[-window:])
    ma_preds.append(pred)
    buffer.append(y_test[i])   # use actual to slide window
ma_forecast = np.array(ma_preds)

# 3. Exponential Smoothing (hand-rolled, alpha=0.3)
alpha = 0.3
es_level = y_train[0]
for val in y_train[1:]:
    es_level = alpha * val + (1 - alpha) * es_level
es_forecast = np.full(n_test, es_level)

# 4. Decomposition + Linear Trend extrapolation
#    a) Estimate trend via polyfit on full training series
trend_coefs = np.polyfit(np.arange(split), y_train, deg=1)
trend_line  = np.polyval(trend_coefs, np.arange(split + n_test))
#    b) Detrend training series
detrended_train = y_train - trend_line[:split]
#    c) Extract weekly seasonality (mean of each weekday over detrended)
season_len = 7
season_pattern = np.array([
    np.mean(detrended_train[i::season_len]) for i in range(season_len)
])
#    d) Forecast = trend + seasonal pattern
dec_forecast = np.array([
    trend_line[split + i] + season_pattern[(split + i) % season_len]
    for i in range(n_test)
])


# ------- Metrics -------
def mae_rmse(actual, pred):
    err  = actual - pred
    mae  = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    return mae, rmse

forecasts = {
    "Naive":               naive_forecast,
    "Moving Avg (w=7)":    ma_forecast,
    "Exp Smooth (a=0.3)":  es_forecast,
    "Decomp+LinTrend":     dec_forecast,
}

print("\nTime-Series Forecasting - Method Comparison")
print("-" * 50)
print(f"{'Method':<22} {'MAE':>8} {'RMSE':>9}")
print("-" * 50)
for name, pred in forecasts.items():
    mae, rmse = mae_rmse(y_test, pred)
    print(f"{name:<22} {mae:>8.2f} {rmse:>9.2f}")
print("-" * 50)

# ------- Plot -------
fig, ax = plt.subplots(figsize=(14, 5))

# Show last 60 days of training + all test
show_train = 60
x_train_show = np.arange(split - show_train, split)
x_test       = np.arange(split, split + n_test)

ax.plot(x_train_show, y_train[-show_train:],
        color="black", lw=1.5, label="Actual (train tail)")
ax.plot(x_test, y_test,
        color="black", lw=1.5, linestyle="--", label="Actual (test)")

colors = ["steelblue", "darkorange", "green", "red"]
for (name, pred), color in zip(forecasts.items(), colors):
    ax.plot(x_test, pred, color=color, lw=1.2, alpha=0.85, label=name)

ax.axvline(split, color="grey", linestyle=":", lw=1)
ax.set_xlabel("Day")
ax.set_ylabel("Sales")
ax.set_title("Classical Time-Series Forecasting (80/20 split)")
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig("ML/outputs/ts_forecasting.png", dpi=100)
plt.close()
print("Plot saved -> ML/outputs/ts_forecasting.png")
