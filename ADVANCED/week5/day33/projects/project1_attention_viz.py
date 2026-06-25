"""
Project: Interactive Attention Heatmap Visualizer
Teaches: visualizing how different sentence pairs produce different attention
         patterns, comparing full vs causal attention side-by-side.
~30 MB RAM, ~1s on CPU
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

SENTENCES = [
    ["The", "cat", "sat", "on", "the", "mat"],
    ["I", "love", "machine", "learning", "models"],
    ["Deep", "neural", "networks", "learn", "features"],
    ["Attention", "is", "all", "you", "need"],
]

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def compute_attention(tokens, d_k=16, seed=None):
    if seed is not None: np.random.seed(seed)
    n = len(tokens)
    Q = np.random.randn(n, d_k); K = np.random.randn(n, d_k); V = np.random.randn(n, d_k)
    scores  = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    return weights

def compute_causal_attention(tokens, d_k=16, seed=None):
    if seed is not None: np.random.seed(seed)
    n = len(tokens)
    Q = np.random.randn(n, d_k); K = np.random.randn(n, d_k); V = np.random.randn(n, d_k)
    scores  = Q @ K.T / np.sqrt(d_k)
    mask    = np.triu(np.ones((n, n), dtype=bool), k=1)
    scores  = np.where(mask, -1e9, scores)
    weights = softmax(scores)
    return weights

def plot_attn(ax, weights, tokens, title, fmt=True):
    im = ax.imshow(weights, cmap="YlOrRd", vmin=0, vmax=weights.max())
    ax.set_xticks(range(len(tokens))); ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    if fmt:
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                ax.text(j, i, f"{weights[i,j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if weights[i,j]>0.4 else "black")
    ax.set_title(title, fontsize=9)
    return im

print("=== Attention Heatmap Visualizer ===\n")
fig, axes = plt.subplots(len(SENTENCES), 2, figsize=(10, 4*len(SENTENCES)))
for row, (tokens, seed) in enumerate(zip(SENTENCES, [0,1,2,3])):
    full_w   = compute_attention(tokens, seed=seed)
    causal_w = compute_causal_attention(tokens, seed=seed)
    fmt = len(tokens) <= 6
    plot_attn(axes[row,0], full_w,   tokens, f"Full Attention: {' '.join(tokens)}", fmt)
    plot_attn(axes[row,1], causal_w, tokens, f"Causal Mask:    {' '.join(tokens)}", fmt)
    # Print entropy stats
    full_ent   = -np.sum(full_w * np.log(full_w+1e-9), axis=-1)
    causal_ent = -np.sum(causal_w * np.log(causal_w+1e-9), axis=-1)
    print(f"Sentence: {' '.join(tokens)}")
    print(f"  Full entropy (per token): {full_ent.round(3)}")
    print(f"  Causal entropy:           {causal_ent.round(3)}")

plt.suptitle("Attention Patterns: Full vs Causal", fontsize=12)
plt.tight_layout(); plt.savefig("attention_viz.png", dpi=85, bbox_inches="tight"); plt.close()
print("\nSaved attention_viz.png")
