# DL Reference — Tiny Transformer Template
# Minimal transformer block from scratch in PyTorch. CPU-friendly.
# d_model=32-64, 1-2 layers, 2-4 heads.
# ~30 MB RAM, <3s on CPU
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ─── 1. POSITIONAL ENCODING ──────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)[:, :d_model//2]
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):   # x: (B, T, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])

# ─── 2. SCALED DOT-PRODUCT ATTENTION ─────────────────────────────────────────
def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q,K,V: (B, heads, T, d_k)"""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2,-1)) / (d_k**0.5)
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V), attn

# ─── 3. MULTI-HEAD ATTENTION ──────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.last_attn = None

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        def reshape(x): return x.view(B,-1,self.n_heads,self.d_k).transpose(1,2)
        Q,K,V = reshape(self.Wq(q)), reshape(self.Wk(k)), reshape(self.Wv(v))
        out, attn = scaled_dot_product_attention(Q,K,V,mask)
        self.last_attn = attn.detach()
        out = out.transpose(1,2).contiguous().view(B,-1,self.n_heads*self.d_k)
        return self.Wo(out)

# ─── 4. TRANSFORMER BLOCK ─────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn  = MultiHeadAttention(d_model, n_heads)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Linear(ff_dim, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.drop(self.attn(x,x,x,mask)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x

# ─── 5. FULL TINY TRANSFORMER ─────────────────────────────────────────────────
class TinyTransformer(nn.Module):
    """
    Tiny transformer for sequence classification or generation.
    For classification: pool=True → outputs (B, output_dim)
    For seq2seq     : pool=False → outputs (B, T, output_dim)
    """
    def __init__(self, vocab_size, d_model=32, n_heads=2, n_layers=2,
                 ff_dim=64, max_len=100, output_dim=2, pool=True, dropout=0.1):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe     = PositionalEncoding(d_model, max_len, dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)])
        self.pool   = pool
        self.head   = nn.Linear(d_model, output_dim)

    def forward(self, x, mask=None):    # x: (B, T) long
        out = self.pe(self.embed(x))
        for blk in self.blocks: out = blk(out, mask)
        out = out.mean(dim=1) if self.pool else out
        return self.head(out)

    def get_attention_weights(self, layer=0):
        """Return attention weights from a specific layer."""
        return self.blocks[layer].attn.last_attn

# ─── 6. CAUSAL MASK HELPER ────────────────────────────────────────────────────
def make_causal_mask(T: int) -> torch.Tensor:
    """Upper-triangular mask for autoregressive decoding."""
    return torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)

# ─── 7. POSITIONAL ENCODING VISUALIZER ───────────────────────────────────────
def visualize_pe(d_model=32, max_len=50, save_path=None):
    pe = PositionalEncoding(d_model, max_len, dropout=0.)
    dummy = torch.zeros(1, max_len, d_model)
    out   = pe(dummy)[0].detach().numpy()
    plt.figure(figsize=(10,4))
    plt.imshow(out.T, cmap="RdBu", aspect="auto")
    plt.colorbar(); plt.xlabel("Position"); plt.ylabel("Encoding dimension")
    plt.title("Positional Encoding Matrix")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=80); plt.close()
    else: plt.show()

# ─── DEMO ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    # Toy task: classify sequences — does sequence sum > threshold?
    VOCAB=10; SEQ_LEN=8; N=500; BATCH=32
    X = torch.randint(0, VOCAB, (N, SEQ_LEN))
    y = (X.float().sum(dim=1) > SEQ_LEN*4.5).long()   # class 1 if sum > threshold

    model = TinyTransformer(VOCAB, d_model=32, n_heads=2, n_layers=2,
                            ff_dim=64, max_len=20, output_dim=2, pool=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()
    split = 400
    for epoch in range(1, 21):
        model.train()
        perm = torch.randperm(split)
        for i in range(0, split, BATCH):
            idx = perm[i:i+BATCH]
            logits = model(X[idx])
            loss   = crit(logits, y[idx])
            optim.zero_grad(); loss.backward(); optim.step()
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(X[split:]).argmax(1)==y[split:]).float().mean()
            print(f"  Epoch {epoch:>2}: val_acc={acc:.3f}")

    visualize_pe(d_model=32, max_len=50, save_path="pe_visualization.png")
    print("Saved pe_visualization.png")
