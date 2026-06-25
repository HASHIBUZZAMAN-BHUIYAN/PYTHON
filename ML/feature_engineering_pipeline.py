"""
What it does: Demonstrates feature engineering and sklearn Pipelines on a synthetic
              messy employee dataset with NaN values and categorical features.
What it teaches: SimpleImputer, OneHotEncoder, StandardScaler, ColumnTransformer,
                 Pipeline, CV accuracy before vs after, coefficient interpretation.
Category: FEATURE ENGINEERING & PIPELINES
RAM estimate: < 100 MB
Time estimate: < 5 seconds
"""

import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("ML/outputs", exist_ok=True)

np.random.seed(42)
n = 500

# --- Generate synthetic messy employee dataset ---
departments = np.random.choice(["Sales", "Eng", "HR", "Ops"], n)
years_exp   = np.random.randint(0, 20, n).astype(float)
age_raw     = 22 + years_exp + np.random.normal(0, 3, n)

# Introduce ~15% NaN in age and ~10% in performance_score
age = age_raw.copy()
age[np.random.choice(n, int(n * 0.15), replace=False)] = np.nan

salary = (30000
          + 2000 * years_exp
          + np.where(departments == "Eng", 10000, 0)
          + np.random.normal(0, 5000, n))

perf_raw = np.random.normal(3.0, 0.8, n).clip(1, 5)
perf = perf_raw.copy()
perf[np.random.choice(n, int(n * 0.10), replace=False)] = np.nan

# Promotion: driven by high perf, high salary, Eng dept
logit = (0.5 * perf_raw
         + 0.00005 * salary
         + np.where(departments == "Eng", 0.5, 0)
         - 2.5)
prob   = 1 / (1 + np.exp(-logit))
promoted = (np.random.uniform(0, 1, n) < prob).astype(int)

df = pd.DataFrame({
    "age":               age,
    "salary":            salary,
    "department":        departments,
    "years_experience":  years_exp,
    "performance_score": perf,
    "promoted":          promoted,
})

# --- BEFORE stats ---
print("=== BEFORE feature engineering ===")
print(f"Shape: {df.shape}")
print(f"NaN counts:\n{df.isnull().sum().to_string()}")
print(f"Promotion rate: {promoted.mean():.2%}")

# --- BEFORE: Naive baseline (drop NaN rows, drop categorical) ---
df_naive = df.dropna().copy()
X_naive  = df_naive[["salary", "years_experience"]].values
y_naive  = df_naive["promoted"].values
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

naive_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr",     LogisticRegression(max_iter=1000, random_state=42)),
])
naive_scores = cross_val_score(naive_pipe, X_naive, y_naive, cv=5, scoring="accuracy")
print(f"\nBaseline CV (drop NaN, 2 numeric features): {naive_scores.mean():.4f} +/- {naive_scores.std():.4f}")

# --- Build full Pipeline ---
numeric_features     = ["age", "salary", "years_experience", "performance_score"]
categorical_features = ["department"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer,      numeric_features),
    ("cat", categorical_transformer,  categorical_features),
])

full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   LogisticRegression(max_iter=1000, random_state=42)),
])

X = df.drop("promoted", axis=1)
y = df["promoted"]
full_scores = cross_val_score(full_pipeline, X, y, cv=5, scoring="accuracy")
print(f"Full pipeline CV (imputed + OHE + 4 features): {full_scores.mean():.4f} +/- {full_scores.std():.4f}")

# --- AFTER stats ---
print("\n=== AFTER feature engineering (pipeline fitted on full data) ===")
full_pipeline.fit(X, y)
preprocessed = full_pipeline.named_steps["preprocessor"].transform(X)
ohe_cats = full_pipeline.named_steps["preprocessor"].named_transformers_["cat"]["ohe"].categories_[0]
all_feature_names = numeric_features + list(ohe_cats)
print(f"Processed feature matrix shape: {preprocessed.shape}")
print(f"Features: {all_feature_names}")

coefs = full_pipeline.named_steps["classifier"].coef_[0]
print("\nLogistic Regression Coefficients (after scaling):")
print("-" * 45)
for fname, coef in sorted(zip(all_feature_names, coefs), key=lambda x: abs(x[1]), reverse=True):
    direction = "[+]" if coef > 0 else "[-]"
    print(f"  {direction} {fname:<22} {coef:>8.4f}")
print("-" * 45)

# --- Bar chart of coefficients ---
fig, ax = plt.subplots(figsize=(9, 5))
sorted_pairs = sorted(zip(all_feature_names, coefs), key=lambda x: x[1])
fnames_sorted = [p[0] for p in sorted_pairs]
coefs_sorted  = [p[1] for p in sorted_pairs]
colors_bar    = ["red" if c < 0 else "steelblue" for c in coefs_sorted]
ax.barh(fnames_sorted, coefs_sorted, color=colors_bar, alpha=0.75)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Coefficient (after StandardScaler)")
ax.set_title("LogisticRegression Coefficients - Promotion Prediction")
plt.tight_layout()
plt.savefig("ML/outputs/feature_pipeline.png", dpi=100)
plt.close()
print("Plot saved -> ML/outputs/feature_pipeline.png")

# --- Pipeline text representation ---
print("\nPipeline structure:")
print("-" * 45)
print("Pipeline([")
print("  preprocessor: ColumnTransformer([")
print("    num: Pipeline([imputer(median), scaler])")
print("    cat: Pipeline([imputer(most_freq), OneHotEncoder])")
print("  ])")
print("  classifier: LogisticRegression(max_iter=1000)")
print("])")
