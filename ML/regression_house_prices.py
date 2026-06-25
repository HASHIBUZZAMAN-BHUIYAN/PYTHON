"""
What it does: Predicts house prices from synthetic features using three regression models.
What it teaches: LinearRegression, Ridge regularization, RandomForest regression,
                 MAE/RMSE/R2 metrics, actual-vs-predicted scatter plots.
Category: REGRESSION
RAM estimate: < 100 MB
Time estimate: < 5 seconds
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("ML/outputs", exist_ok=True)

# --- Generate synthetic housing data ---
np.random.seed(42)
n = 1000
size_sqft   = np.random.uniform(500, 3000, n)
bedrooms    = np.random.randint(1, 6, n).astype(float)
age_years   = np.random.uniform(0, 50, n)
distance_km = np.random.uniform(1, 30, n)
noise       = np.random.normal(0, 15000, n)

price = (150 * size_sqft
         + 20000 * bedrooms
         - 1000 * age_years
         - 5000 * distance_km
         + noise)

df = pd.DataFrame({
    "size_sqft":   size_sqft,
    "bedrooms":    bedrooms,
    "age_years":   age_years,
    "distance_km": distance_km,
    "price":       price,
})

X = df.drop("price", axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Models ---
models = {
    "LinearRegression": LinearRegression(),
    "Ridge(alpha=10)":  Ridge(alpha=10),
    "RandomForest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "preds": preds}

# --- Print comparison table ---
print("House Price Regression - Model Comparison")
print("-" * 60)
header = f"{'Model':<22} {'MAE':>10} {'RMSE':>12} {'R2':>8}"
print(header)
print("-" * 60)
best_name, best_r2 = None, -999
for name, m in results.items():
    print(f"{name:<22} {m['MAE']:>10,.0f} {m['RMSE']:>12,.0f} {m['R2']:>8.4f}")
    if m["R2"] > best_r2:
        best_r2   = m["R2"]
        best_name = name
print("-" * 60)
print(f"Best model: {best_name} (R2={best_r2:.4f})")

# --- Plot: actual vs predicted for best model ---
best_preds = results[best_name]["preds"]

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, best_preds, alpha=0.4, s=20, color="steelblue", label="Predictions")
lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
ax.set_xlabel("Actual Price ($)")
ax.set_ylabel("Predicted Price ($)")
ax.set_title(f"Actual vs Predicted - {best_name}")
ax.legend()
plt.tight_layout()
plt.savefig("ML/outputs/regression_house.png", dpi=100)
plt.close()
print("Plot saved -> ML/outputs/regression_house.png")
