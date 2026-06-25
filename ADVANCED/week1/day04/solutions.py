# Advanced Day 04 — Solutions
import numpy as np
from scipy import stats
np.random.seed(0)

# Exercise 1
data = np.array([4,7,13,2,1,8,6,3,14,99,5,9,2,7,4,3,8,6,5,11], dtype=float)
q1, q3 = np.percentile(data, [25, 75]); iqr = q3 - q1
outliers = data[(data < q1-1.5*iqr) | (data > q3+1.5*iqr)]
print(f"Mean={data.mean():.2f} Median={np.median(data):.2f} Std={data.std():.2f}")
print(f"IQR={iqr:.2f} Outliers={outliers}")

# Exercise 2
a = np.random.normal(50, 10, 10000)
for k in [1,2,3]:
    frac = ((a > 50-k*10) & (a < 50+k*10)).mean()
    print(f"Within {k}σ: {frac*100:.1f}%")
lam = 1000 * 0.02
print(f"P(≤30 fail) = {stats.poisson.cdf(30, lam):.4f}")

# Exercise 3
import matplotlib.pyplot as plt
x = np.random.uniform(0,100,100); y = 2*x + np.random.normal(0,10,100); z = np.random.uniform(0,100,100)
vars_ = {"x":x,"y":y,"z":z}
fig, axes = plt.subplots(3,3,figsize=(8,8))
pairs = list(vars_.items())
for i,(n1,v1) in enumerate(pairs):
    for j,(n2,v2) in enumerate(pairs):
        ax = axes[i][j]
        if i==j: ax.hist(v1,bins=15); ax.set_xlabel(n1)
        else:
            r,_ = stats.pearsonr(v1,v2)
            ax.scatter(v2,v1,s=10,alpha=0.5); ax.set_xlabel(n2); ax.set_ylabel(n1)
            ax.set_title(f"r={r:.2f}",fontsize=8)
plt.tight_layout(); plt.savefig("corr_matrix.png",dpi=72); plt.close()
print("Saved corr_matrix.png")

# Exercise 4
def run_ab_test():
    a_conv = np.random.binomial(500, 0.05); b_conv = np.random.binomial(500, 0.07)
    table = np.array([[a_conv, 500-a_conv],[b_conv, 500-b_conv]])
    _, p, _, _ = stats.chi2_contingency(table)
    return p < 0.05
rejections = sum(run_ab_test() for _ in range(100))
print(f"Correctly rejected H0 in {rejections}/100 trials (power estimate)")

# Exercise 5
sample = np.array([23,31,18,25,29,22,35,27,19,30,24,28,21,33,26])
boot_means = [np.random.choice(sample, len(sample), replace=True).mean() for _ in range(10000)]
lo, hi = np.percentile(boot_means, [2.5, 97.5])
print(f"Bootstrap 95% CI for mean: [{lo:.2f}, {hi:.2f}]")
