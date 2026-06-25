# Advanced Day 33 — Attention Mechanism
# ~60 MB RAM, ~3s on CPU

print("""
=== Attention Mechanism — Day 33 ===

Attention allows a model to selectively focus on relevant parts of input.

Scaled Dot-Product Attention:
  Inputs: Q (query), K (key), V (value)  — all shape [seq, d_k]
  Step 1: scores = Q @ K.T / sqrt(d_k)      [seq, seq]
  Step 2: weights = softmax(scores)           [seq, seq]  — each row sums to 1
  Step 3: output = weights @ V               [seq, d_v]

Multi-Head Attention:
  Run h attention heads in parallel with different learned projections,
  then concatenate and project.
  Allows the model to attend to different aspects simultaneously.
""")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── 1. NUMPY IMPLEMENTATION ─────────────────────────────────────────────────
print("=== 1. Scaled Dot-Product Attention (NumPy) ===")

def softmax(x, axis=-1):
    e_x = np.exp(x - x.max(axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def scaled_dot_product(Q, K, V, mask=None):
    d_k    = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)          # [seq, seq]
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    weights = softmax(scores, axis=-1)         # [seq, seq]
    output  = weights @ V                      # [seq, d_v]
    return output, weights

# Demo with 4 tokens, d_model=8
np.random.seed(42)
seq_len, d_k = 4, 8
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)
out, weights = scaled_dot_product(Q, K, V)
print(f"  Q shape: {Q.shape}  Output shape: {out.shape}")
print(f"  Attention weights (rows sum to 1):")
for i, row in enumerate(weights):
    print(f"    Token {i}: {row.round(3)}  sum={row.sum():.4f}")

# ─── 2. CAUSAL MASK (for language modeling) ───────────────────────────────────
print("\n=== 2. Causal Mask (no looking ahead) ===")
seq = 5
mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)  # upper triangle
Q2=np.random.randn(seq,d_k); K2=np.random.randn(seq,d_k); V2=np.random.randn(seq,d_k)
_, w2 = scaled_dot_product(Q2, K2, V2, mask=mask)
print("  Causal weights (each token only attends to previous):")
for i, row in enumerate(w2):
    print(f"    Token {i}: {row.round(3)}")

# ─── 3. PYTORCH MULTI-HEAD ATTENTION ─────────────────────────────────────────
print("\n=== 3. Multi-Head Attention (PyTorch) ===")
torch.manual_seed(42)
d_model, n_heads = 16, 4
mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

seq_len = 6
x = torch.randn(1, seq_len, d_model)  # [batch, seq, d_model]
out, attn_weights = mha(x, x, x)
print(f"  Input:  {x.shape}")
print(f"  Output: {out.shape}")
print(f"  Attn weights: {attn_weights.shape}  (averaged over heads)")

# ─── 4. VISUALIZATION ─────────────────────────────────────────────────────────
TOKENS = ["The", "cat", "sat", "on", "mat"]
seq_l  = len(TOKENS)
Q3=np.random.randn(seq_l,d_k); K3=np.random.randn(seq_l,d_k); V3=np.random.randn(seq_l,d_k)
_, w3 = scaled_dot_product(Q3, K3, V3)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im0 = axes[0].imshow(w3, cmap="viridis", vmin=0)
axes[0].set_xticks(range(seq_l)); axes[0].set_yticks(range(seq_l))
axes[0].set_xticklabels(TOKENS, rotation=35); axes[0].set_yticklabels(TOKENS)
axes[0].set_title("Attention Weights (full)"); plt.colorbar(im0, ax=axes[0])
for i in range(seq_l):
    for j in range(seq_l):
        axes[0].text(j,i,f"{w3[i,j]:.2f}",ha="center",va="center",fontsize=8,
                     color="white" if w3[i,j]<0.35 else "black")

causal_m=np.triu(np.ones((seq_l,seq_l),dtype=bool),k=1)
_,w4=scaled_dot_product(Q3,K3,V3,mask=causal_m)
im1=axes[1].imshow(w4,cmap="Blues",vmin=0)
axes[1].set_xticks(range(seq_l)); axes[1].set_yticks(range(seq_l))
axes[1].set_xticklabels(TOKENS,rotation=35); axes[1].set_yticklabels(TOKENS)
axes[1].set_title("Causal Attention (masked)"); plt.colorbar(im1,ax=axes[1])

plt.suptitle("Scaled Dot-Product Attention Visualization", fontsize=11)
plt.tight_layout(); plt.savefig("attention_lesson.png", dpi=85); plt.close()
print("Saved attention_lesson.png")

print("""
=== Key Takeaways ===
  d_k scaling: dividing by sqrt(d_k) prevents softmax saturation
  Self-attention: Q=K=V come from the same sequence (like BERT)
  Cross-attention: Q from one seq, K/V from another (like in Seq2Seq decoder)
  Causal mask: ensures each position only attends to earlier positions
""")
