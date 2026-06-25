# Advanced Day 03 — Data Visualization
# ~80 MB RAM, <5s on CPU

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

# ─── 1. LINE PLOT ────────────────────────────────────────────────────────────
x = np.linspace(0, 2*np.pi, 100)
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(x, np.sin(x), label="sin", color="blue",  linewidth=2)
ax.plot(x, np.cos(x), label="cos", color="orange", linewidth=2, linestyle="--")
ax.set_title("Sin and Cos"); ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("01_line.png", dpi=72); plt.close()

# ─── 2. SCATTER PLOT ─────────────────────────────────────────────────────────
n = 100
x = np.random.randn(n); y = x * 1.5 + np.random.randn(n) * 0.5
colors = np.where(y > 0, "steelblue", "tomato")
fig, ax = plt.subplots(figsize=(5,4))
ax.scatter(x, y, c=colors, alpha=0.7, edgecolors="white", s=40)
ax.set_title("Scatter Plot"); ax.set_xlabel("x"); ax.set_ylabel("y")
plt.tight_layout(); plt.savefig("02_scatter.png", dpi=72); plt.close()

# ─── 3. BAR CHART ────────────────────────────────────────────────────────────
categories = ["A","B","C","D","E"]
values = [23, 45, 56, 12, 38]
errors = [3,  5,  7,  2,  4]
fig, ax = plt.subplots(figsize=(6,4))
bars = ax.bar(categories, values, yerr=errors, capsize=5, color="steelblue", alpha=0.8)
ax.bar_label(bars, padding=3)
ax.set_title("Bar Chart with Error Bars"); ax.set_ylabel("Value")
plt.tight_layout(); plt.savefig("03_bar.png", dpi=72); plt.close()

# ─── 4. HISTOGRAM ────────────────────────────────────────────────────────────
data = np.concatenate([np.random.normal(0,1,500), np.random.normal(4,0.5,300)])
fig, ax = plt.subplots(figsize=(6,4))
ax.hist(data, bins=40, density=True, color="green", alpha=0.7, edgecolor="white")
ax.set_title("Histogram"); ax.set_xlabel("Value"); ax.set_ylabel("Density")
plt.tight_layout(); plt.savefig("04_histogram.png", dpi=72); plt.close()

# ─── 5. PIE CHART ────────────────────────────────────────────────────────────
labels = ["Python","Java","JavaScript","C++","Other"]
sizes  = [34, 21, 26, 11, 8]
explode = (0.1, 0, 0, 0, 0)
fig, ax = plt.subplots(figsize=(5,5))
ax.pie(sizes, explode=explode, labels=labels, autopct="%1.1f%%", startangle=140)
ax.set_title("Language Popularity")
plt.tight_layout(); plt.savefig("05_pie.png", dpi=72); plt.close()

# ─── 6. HEATMAP (manual — no seaborn) ───────────────────────────────────────
corr_data = np.random.randn(50, 5)
corr_matrix = np.corrcoef(corr_data.T)
fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
labels = [f"F{i}" for i in range(5)]
ax.set_xticks(range(5)); ax.set_xticklabels(labels)
ax.set_yticks(range(5)); ax.set_yticklabels(labels)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{corr_matrix[i,j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Correlation Heatmap")
plt.tight_layout(); plt.savefig("06_heatmap.png", dpi=72); plt.close()

# ─── 7. SUBPLOTS GRID ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Multiple Plots in One Figure")

axes[0,0].plot(np.random.cumsum(np.random.randn(100))); axes[0,0].set_title("Random Walk")
axes[0,1].hist(np.random.randn(500), bins=30, color="salmon"); axes[0,1].set_title("Normal dist")
axes[1,0].scatter(np.random.rand(50), np.random.rand(50), s=100, c="teal"); axes[1,0].set_title("Scatter")
x2 = np.linspace(0,10,200)
axes[1,1].fill_between(x2, np.sin(x2), alpha=0.4); axes[1,1].set_title("Fill between")
plt.tight_layout(); plt.savefig("07_subplots.png", dpi=72); plt.close()

print("Saved 7 plot files (01-07). Run this script to regenerate them.")
print("Plots: line, scatter, bar, histogram, pie, heatmap, subplots.")
