# Advanced Day 03 — Solutions
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# Exercise 1
fig, ax = plt.subplots(figsize=(8,4))
for i, color in enumerate(["blue","red","green"]):
    walk = np.cumsum(np.random.randn(200))
    ax.plot(walk, color=color, label=f"Walk {i+1}", alpha=0.8)
ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax.legend(); ax.set_title("3 Random Walks")
plt.tight_layout(); plt.savefig("ex1.png", dpi=72); plt.close()

# Exercise 2
subjects = ["Math","Science","English","History"]
class_a, class_b = [78,82,70,88], [85,77,92,74]
x = np.arange(len(subjects)); w = 0.35
fig, ax = plt.subplots(figsize=(7,4))
ax.bar(x-w/2, class_a, w, label="Class A"); ax.bar(x+w/2, class_b, w, label="Class B")
ax.set_xticks(x); ax.set_xticklabels(subjects)
ax.set_ylabel("Score"); ax.legend(); ax.set_title("Class Comparison")
plt.tight_layout(); plt.savefig("ex2.png", dpi=72); plt.close()

# Exercise 3
x, y = np.random.randn(200), np.random.randn(200)
d = np.sqrt(x**2 + y**2)
fig, ax = plt.subplots(figsize=(5,5))
sc = ax.scatter(x, y, c=d, cmap="plasma", s=30, alpha=0.8)
plt.colorbar(sc, ax=ax, label="Distance from origin")
ax.set_title("Distance-colored scatter")
plt.tight_layout(); plt.savefig("ex3.png", dpi=72); plt.close()

# Exercise 4
x = np.linspace(-3, 3, 200)
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(x, x**2, "blue", linewidth=2)
ax.annotate("Minimum", xy=(0,0), xytext=(1.5, 2),
            arrowprops=dict(arrowstyle="->", color="red"), color="red", fontsize=11)
ax.fill_between(x, x**2, where=(x>1), alpha=0.2, color="red", label="x>1")
ax.set_title("y = x²"); ax.legend()
plt.tight_layout(); plt.savefig("ex4.png", dpi=72); plt.close()

# Exercise 5
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except:
    plt.style.use("ggplot")
fig, ax = plt.subplots(figsize=(8,4))
for i, color in enumerate(["blue","red","green"]):
    ax.plot(np.cumsum(np.random.randn(200)), label=f"Walk {i+1}", alpha=0.8)
ax.axhline(0, color="k", linestyle="--"); ax.legend(); ax.set_title("Styled random walks")
plt.tight_layout(); plt.savefig("ex5.png", dpi=72); plt.close()
plt.style.use("default")
print("Saved ex1-ex5.png")
