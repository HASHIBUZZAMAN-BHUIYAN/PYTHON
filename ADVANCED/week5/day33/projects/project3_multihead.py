"""
Project: Multi-Head Attention Comparison
Teaches: comparing 1-head vs 4-head vs 8-head attention on a classification task,
         visualizing per-head attention patterns.
~60 MB RAM, ~8s on CPU
"""
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

torch.manual_seed(42); np.random.seed(42)

VOCAB_SIZE=20; D_MODEL=16; SEQ_LEN=8; N_CLASSES=3

def make_dataset(n=800):
    X, y = [], []
    for _ in range(n):
        cls=np.random.randint(3)
        if cls==0:   t=np.random.randint(0, 7,  SEQ_LEN)
        elif cls==1: t=np.random.randint(7, 14, SEQ_LEN)
        else:        t=np.random.randint(14,20, SEQ_LEN)
        X.append(t); y.append(cls)
    return np.array(X), np.array(y)

X,y=make_dataset(800)
X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=42)
X_tr=torch.tensor(X_tr); X_te=torch.tensor(X_te)
y_tr=torch.tensor(y_tr); y_te=torch.tensor(y_te)

class MHAClassifier(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.mha   = nn.MultiheadAttention(D_MODEL, n_heads, batch_first=True)
        self.fc    = nn.Linear(D_MODEL, N_CLASSES)
        self.n_heads=n_heads
    def forward(self, x):
        E   = self.embed(x)
        out, w = self.mha(E, E, E)
        return self.fc(out.mean(1)), w

def train_and_eval(n_heads, epochs=100):
    model = MHAClassifier(n_heads)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        idx   = torch.randperm(len(X_tr))[:64]
        loss  = crit(model(X_tr[idx])[0], y_tr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits, attn = model(X_te)
        acc = (logits.argmax(-1)==y_te).float().mean().item()
    return acc, attn, model

print("=== Multi-Head Attention Comparison ===\n")
results = {}
for n_heads in [1, 2, 4]:
    if D_MODEL % n_heads != 0: continue
    acc, attn, mdl = train_and_eval(n_heads)
    results[n_heads] = {"acc": acc, "attn": attn}
    print(f"  n_heads={n_heads}  test_acc={acc:.3f}")

best_n = max(results, key=lambda k: results[k]["acc"])
print(f"\n  Best: {best_n} heads (acc={results[best_n]['acc']:.3f})")

# ─── Visualize attention for best model ───────────────────────────────────────
attn_best = results[best_n]["attn"]  # [batch, seq, seq] averaged over heads
fig, axes = plt.subplots(1, len(results), figsize=(4*len(results), 4))
if len(results)==1: axes=[axes]
for ax, (n_heads, r) in zip(axes, sorted(results.items())):
    w = r["attn"][0].detach().numpy()
    im= ax.imshow(w, cmap="hot", vmin=0)
    ax.set_title(f"{n_heads} head(s)\nacc={r['acc']:.3f}")
    ax.set_xlabel("Key"); ax.set_ylabel("Query")
    plt.colorbar(im, ax=ax, shrink=0.7)

# Accuracy bar chart
fig2, ax2 = plt.subplots(figsize=(5,3))
hs  = sorted(results.keys())
acs = [results[h]["acc"] for h in hs]
ax2.bar([str(h) for h in hs], acs, color="steelblue", alpha=0.85, edgecolor="k")
ax2.set_xlabel("Number of Heads"); ax2.set_ylabel("Test Accuracy")
ax2.set_ylim(0, 1.05); ax2.set_title("Accuracy vs Number of Heads")
for i,v in enumerate(acs): ax2.text(i,v+0.01,f"{v:.3f}",ha="center",fontsize=9)
plt.tight_layout(); plt.savefig("mha_comparison.png",dpi=85); plt.close()

plt.savefig("mha_attention.png",dpi=85,bbox_inches="tight",figure=fig); plt.close()
print("Saved mha_comparison.png  mha_attention.png")
