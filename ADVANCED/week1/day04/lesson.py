# Advanced Day 04 — Statistics & Probability for ML
# ~60 MB RAM, <5s on CPU

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# ─── 1. DESCRIPTIVE STATISTICS ───────────────────────────────────────────────
data = np.array([14,18,11,13,6,8,2,5,11,6,14,8,11,5,3,7,12,9,14,11])
print("=== Descriptive Stats ===")
print(f"N        : {len(data)}")
print(f"Mean     : {data.mean():.2f}")
print(f"Median   : {np.median(data):.2f}")
print(f"Mode     : {stats.mode(data, keepdims=True).mode[0]}")
print(f"Variance : {data.var():.2f}")
print(f"Std Dev  : {data.std():.2f}")
print(f"Skewness : {stats.skew(data):.3f}")
print(f"Kurtosis : {stats.kurtosis(data):.3f}")
print(f"IQR      : {np.percentile(data,75) - np.percentile(data,25):.2f}")
print(f"Range    : {data.max() - data.min()}")

# ─── 2. PROBABILITY DISTRIBUTIONS ───────────────────────────────────────────
print("\n=== Distributions ===")

mu, sigma = 170, 10
x = np.linspace(130, 210, 300)
pdf_normal = stats.norm.pdf(x, mu, sigma)
print(f"P(height > 180 | N({mu},{sigma})) = {1 - stats.norm.cdf(180, mu, sigma):.4f}")

n_flips, p = 10, 0.5
k = np.arange(0, 11)
pmf_binom = stats.binom.pmf(k, n_flips, p)
print(f"P(exactly 7 heads in 10 flips) = {stats.binom.pmf(7, 10, 0.5):.4f}")

lam = 3
print(f"P(>=5 events | Poisson(3)) = {1 - stats.poisson.cdf(4, lam):.4f}")

# ─── 3. CORRELATION ──────────────────────────────────────────────────────────
print("\n=== Correlation ===")
height = np.random.normal(170, 10, 50)
weight = 0.4 * height + np.random.normal(0, 5, 50)   # correlated
foot   = np.random.normal(26, 2, 50)                   # uncorrelated

pearson_hw, p_hw = stats.pearsonr(height, weight)
pearson_hf, p_hf = stats.pearsonr(height, foot)
print(f"Height vs Weight: r={pearson_hw:.3f}, p={p_hw:.4f}")
print(f"Height vs Foot:   r={pearson_hf:.3f}, p={p_hf:.4f}")

# Spearman (rank-based, more robust to outliers)
spearman_hw, _ = stats.spearmanr(height, weight)
print(f"Spearman Height vs Weight: ρ={spearman_hw:.3f}")

# ─── 4. HYPOTHESIS TESTING ───────────────────────────────────────────────────
print("\n=== Hypothesis Testing ===")
# H0: groups have same mean. Ha: different means.
group_a = np.random.normal(68, 5, 30)
group_b = np.random.normal(72, 5, 30)

t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value    : {p_value:.4f}")
alpha = 0.05
if p_value < alpha:
    print(f"Reject H0 (p < {alpha}) — groups are significantly different")
else:
    print(f"Fail to reject H0 (p >= {alpha})")

t_one, p_one = stats.ttest_1samp(data, popmean=10)
print(f"\nOne-sample t-test (H0: mean=10): t={t_one:.3f}, p={p_one:.4f}")

# ─── 5. CENTRAL LIMIT THEOREM ────────────────────────────────────────────────
print("\n=== Central Limit Theorem ===")
# Sample from a non-normal distribution many times; sample means ≈ normal
pop = np.random.exponential(scale=2, size=100000)
sample_means = [np.random.choice(pop, 30, replace=False).mean() for _ in range(1000)]
print(f"Population mean : {pop.mean():.3f}")
print(f"Sample means avg: {np.mean(sample_means):.3f}")
print(f"Sample means std: {np.std(sample_means):.4f}  (≈ sigma/sqrt(n) = {pop.std()/np.sqrt(30):.4f})")
