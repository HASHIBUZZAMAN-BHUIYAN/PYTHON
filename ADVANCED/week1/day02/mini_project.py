# Advanced Day 02 Mini-Project — Student Performance Analysis
# ~60 MB RAM, <5s on CPU

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

# ─── Generate synthetic student dataset ──────────────────────────────────────
n = 200
subjects = ["Math", "Science", "English", "History", "Art"]

data = {"StudentID": range(1, n+1),
        "Gender": np.random.choice(["M","F"], n),
        "Grade": np.random.choice(["9","10","11","12"], n)}
for sub in subjects:
    base = np.random.randint(50, 100, n)
    data[sub] = np.clip(base + np.random.randint(-10, 10, n), 0, 100)

df = pd.DataFrame(data)
df["Average"] = df[subjects].mean(axis=1).round(1)
df["Pass"]    = df["Average"] >= 60

# ─── Analysis ────────────────────────────────────────────────────────────────
print("=== Student Performance Report ===\n")
print(f"Total students : {len(df)}")
print(f"Pass rate      : {df['Pass'].mean()*100:.1f}%")
print(f"\nAverage scores by grade:")
print(df.groupby("Grade")[subjects].mean().round(1))
print(f"\nAverage scores by gender:")
print(df.groupby("Gender")[subjects].mean().round(1))
print(f"\nTop 5 students:")
print(df.nlargest(5, "Average")[["StudentID","Grade","Average"]])
print(f"\nSubject with highest avg: {df[subjects].mean().idxmax()}")
print(f"Subject with lowest avg : {df[subjects].mean().idxmin()}")

# ─── Visualise ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Student Performance Analysis")

axes[0].hist(df["Average"], bins=20, color="steelblue", edgecolor="white")
axes[0].axvline(60, color="red", linestyle="--", label="Pass threshold")
axes[0].set_title("Distribution of Averages")
axes[0].set_xlabel("Score"); axes[0].legend()

df[subjects].boxplot(ax=axes[1])
axes[1].set_title("Score Distribution per Subject")
axes[1].tick_params(axis="x", rotation=30)

pass_rate = df.groupby("Grade")["Pass"].mean() * 100
pass_rate.sort_index().plot(kind="bar", ax=axes[2], color="green", edgecolor="white")
axes[2].set_title("Pass Rate by Grade")
axes[2].set_ylabel("%"); axes[2].set_ylim(0,100)
axes[2].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig("student_performance.png", dpi=80)
print("\nSaved student_performance.png")
plt.show()
