# Advanced Day 04 Mini-Project — Statistical Report on Synthetic Health Data
# ~70 MB RAM, <5s on CPU

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# ─── Generate synthetic health dataset ───────────────────────────────────────
n = 300
age    = np.random.randint(20, 70, n)
height = np.random.normal(168, 10, n)
weight = 0.45 * height + np.random.normal(0, 8, n)
bmi    = weight / (height/100)**2

# Systolic blood pressure: older → slightly higher
sbp = 110 + 0.5*age + np.random.normal(0, 10, n)
sbp = np.clip(sbp, 90, 180)

smoker      = np.random.choice([0,1], n, p=[0.7, 0.3])
cholesterol = 180 + 20*smoker + np.random.normal(0, 15, n)
has_disease = ((sbp > 140) | (bmi > 30) | (cholesterol > 220)).astype(int)

df = pd.DataFrame({"age":age,"height":height,"weight":weight,"bmi":bmi,
                   "sbp":sbp,"smoker":smoker,"cholesterol":cholesterol,
                   "disease":has_disease})

# ─── Statistical analysis ─────────────────────────────────────────────────────
print("=" * 50)
print("SYNTHETIC HEALTH DATA — STATISTICAL REPORT")
print("=" * 50)
print(f"\nN = {n}")
print(df.describe().round(2))

d1 = df[df["disease"]==1]["sbp"]
d0 = df[df["disease"]==0]["sbp"]
t, p = stats.ttest_ind(d1, d0)
print(f"\nSBP: disease={d1.mean():.1f}, no-disease={d0.mean():.1f}")
print(f"t-test: t={t:.3f}, p={p:.4e} → {'Significant' if p<0.05 else 'NS'}")

for col in ["age","bmi","sbp","cholesterol"]:
    r, pv = stats.pointbiserialr(df["disease"], df[col])
    print(f"Correlation ({col} vs disease): r={r:.3f}, p={pv:.4f}")

# ─── Visualise ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Health Data Statistical Analysis", fontsize=14)

axes[0,0].hist(df["bmi"], bins=25, color="steelblue", edgecolor="white")
axes[0,0].set_title("BMI Distribution"); axes[0,0].axvline(25,color="orange",linestyle="--",label="Normal")
axes[0,0].axvline(30,color="red",linestyle="--",label="Obese"); axes[0,0].legend(fontsize=8)

axes[0,1].scatter(df["age"],df["sbp"], c=df["disease"], cmap="coolwarm", alpha=0.4, s=15)
axes[0,1].set_title("Age vs SBP (red=disease)"); axes[0,1].set_xlabel("Age"); axes[0,1].set_ylabel("SBP")

bp_data = [d0.values, d1.values]
bp = axes[0,2].boxplot(bp_data, labels=["No disease","Disease"], patch_artist=True)
for patch, color in zip(bp["boxes"],["lightblue","salmon"]): patch.set_facecolor(color)
axes[0,2].set_title("SBP by Disease Status"); axes[0,2].set_ylabel("SBP")

axes[1,0].hist([df[df["smoker"]==0]["cholesterol"], df[df["smoker"]==1]["cholesterol"]],
               bins=20, label=["Non-smoker","Smoker"], alpha=0.6, color=["blue","red"])
axes[1,0].set_title("Cholesterol by Smoking Status"); axes[1,0].legend()

numeric_cols = ["age","bmi","sbp","cholesterol","disease"]
corr = df[numeric_cols].corr()
im = axes[1,1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=axes[1,1])
axes[1,1].set_xticks(range(len(numeric_cols))); axes[1,1].set_xticklabels(numeric_cols, rotation=45, fontsize=8)
axes[1,1].set_yticks(range(len(numeric_cols))); axes[1,1].set_yticklabels(numeric_cols, fontsize=8)
axes[1,1].set_title("Correlation Heatmap")
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        axes[1,1].text(j,i,f"{corr.iloc[i,j]:.2f}",ha="center",va="center",fontsize=7)

disease_rate = df.groupby(pd.cut(df["age"],[20,30,40,50,60,70]))["disease"].mean()*100
disease_rate.plot(kind="bar", ax=axes[1,2], color="salmon", edgecolor="white")
axes[1,2].set_title("Disease Rate by Age Group"); axes[1,2].set_ylabel("%")
axes[1,2].tick_params(axis="x", rotation=30)

plt.tight_layout(); plt.savefig("health_report.png", dpi=80, bbox_inches="tight")
print("\nSaved health_report.png")
plt.show()
