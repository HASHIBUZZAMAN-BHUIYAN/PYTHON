# Advanced Day 34 — Transformer Basics
# ~80 MB RAM, ~10s on CPU

print("""
=== Transformer Architecture — Day 34 ===

The Transformer (Vaswani et al., 2017) replaced RNN/LSTM with pure attention.

Building blocks:
  1. Input Embedding + Positional Encoding
  2. Transformer Block = MultiHeadAttn → Add&Norm → FFN → Add&Norm
  3. Stack N blocks, then task head (classify, generate, etc.)

Positional Encoding (sinusoidal):
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
  Allows model to use position info without sequential processing.
""")

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ─── 1. POSITIONAL ENCODING ──────────────────────────────────────────────────
print("=== 1. Positional Encoding ===")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos= torch.arange(max_len).unsqueeze(1).float()
        div= torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

pe_layer = PositionalEncoding(d_model=16, max_len=20)
x_test   = torch.zeros(1, 10, 16)
x_pe     = pe_layer(x_test)
print(f"  Input shape:  {x_test.shape}")
print(f"  After PE:     {x_pe.shape}")
print(f"  PE matrix (first 4 positions, first 6 dims):")
print(pe_layer.pe[0, :4, :6].numpy().round(3))

# ─── 2. TRANSFORMER BLOCK ────────────────────────────────────────────────────
print("\n=== 2. Transformer Block ===")

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn   = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.norm1  = nn.LayerNorm(d_model)
        self.ff     = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm2  = nn.LayerNorm(d_model)
        self.dropout= nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

block = TransformerBlock(d_model=16, n_heads=4, ff_dim=32)
x_in  = torch.randn(2, 8, 16)  # [batch, seq, d_model]
x_out = block(x_in)
print(f"  Input:  {x_in.shape}  →  Output: {x_out.shape}")
print(f"  Param count: {sum(p.numel() for p in block.parameters()):,}")

# ─── 3. TINY TRANSFORMER CLASSIFIER ──────────────────────────────────────────
print("\n=== 3. TinyTransformer Classifier ===")

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=16, n_heads=4, n_layers=2,
                 ff_dim=32, max_len=50, n_classes=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe    = PositionalEncoding(d_model, max_len)
        self.blocks= nn.Sequential(*[TransformerBlock(d_model,n_heads,ff_dim) for _ in range(n_layers)])
        self.fc    = nn.Linear(d_model, n_classes)

    def forward(self, x):
        e   = self.pe(self.embed(x))  # [B, S, D]
        out = e
        for block in self.blocks:
            out = block(out)
        return self.fc(out.mean(1))   # mean pool → classify

model = TinyTransformer(vocab_size=20, d_model=16, n_heads=4, n_layers=2)
x_dummy = torch.randint(0, 20, (4, 8))
logits  = model(x_dummy)
print(f"  Input tokens: {x_dummy.shape}  → Logits: {logits.shape}")
print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

# ─── 4. TRAINING QUICK DEMO ─────────────────────────────────────────────────
print("\n=== 4. Quick Training Demo ===")
import torch.optim as optim

X = torch.cat([
    torch.randint(0,7,(100,8)), torch.randint(7,14,(100,8)), torch.randint(14,20,(100,8))
])
y = torch.cat([torch.zeros(100), torch.ones(100), torch.full((100,),2)]).long()
idx_perm = torch.randperm(300)
X,y = X[idx_perm], y[idx_perm]

crit = nn.CrossEntropyLoss()
opt  = optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(100):
    i = torch.randperm(300)[:64]
    loss = crit(model(X[i]), y[i])
    opt.zero_grad(); loss.backward(); opt.step()
    if (epoch+1)%25==0:
        with torch.no_grad(): acc=(model(X).argmax(-1)==y).float().mean()
        print(f"  Epoch {epoch+1}  loss={loss.item():.4f}  acc={acc:.3f}")

# ─── 5. PE VISUALIZATION ────────────────────────────────────────────────────
pe_matrix = pe_layer.pe[0, :20, :].numpy()
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(pe_matrix.T, cmap="RdBu", aspect="auto")
ax.set_xlabel("Position"); ax.set_ylabel("Dimension")
ax.set_title("Sinusoidal Positional Encoding"); plt.colorbar(im, ax=ax)
plt.tight_layout(); plt.savefig("transformer_lesson.png", dpi=85); plt.close()
print("\nSaved transformer_lesson.png")
print("""
=== Key Takeaways ===
  Pre-norm (Norm before attn) is more stable than post-norm
  GELU > ReLU in transformers (smoother gradient)
  Residual connections are critical: without them gradients vanish in deep stacks
  d_model must be divisible by n_heads
""")
