"""
Project: Self-Attention Sequence Classifier
Teaches: using self-attention as a feature extractor for sequence classification,
         pooling attended representations, training end-to-end.
~50 MB RAM, ~5s on CPU
"""
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

torch.manual_seed(42); np.random.seed(42)

VOCAB_SIZE = 20; D_MODEL = 16; SEQ_LEN = 8; N_CLASSES = 3

# ─── Synthetic: 3 classes distinguished by which tokens appear ───────────────
def make_dataset(n=600):
    X, y = [], []
    for _ in range(n):
        cls = np.random.randint(3)
        # Class 0: first tokens are low indices (0-6)
        # Class 1: middle tokens (7-13)
        # Class 2: high tokens (14-19)
        if cls == 0:   tokens = np.random.randint(0, 7,  SEQ_LEN)
        elif cls == 1: tokens = np.random.randint(7, 14, SEQ_LEN)
        else:          tokens = np.random.randint(14, 20, SEQ_LEN)
        X.append(tokens); y.append(cls)
    return np.array(X), np.array(y)

X, y = make_dataset(600)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
X_tr=torch.tensor(X_tr); X_te=torch.tensor(X_te)
y_tr=torch.tensor(y_tr); y_te=torch.tensor(y_te)

class SelfAttnClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.Wq    = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wk    = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wv    = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.fc    = nn.Linear(D_MODEL, N_CLASSES)

    def forward(self, x):
        E = self.embed(x)                           # [B, S, D]
        Q = self.Wq(E); K = self.Wk(E); V = self.Wv(E)
        scores  = Q @ K.transpose(-2,-1) / D_MODEL**0.5  # [B, S, S]
        weights = F.softmax(scores, dim=-1)
        out     = weights @ V                       # [B, S, D]
        pooled  = out.mean(dim=1)                   # [B, D]  mean pool
        return self.fc(pooled), weights

model = SelfAttnClassifier()
opt   = optim.Adam(model.parameters(), lr=1e-3)
crit  = nn.CrossEntropyLoss()

EPOCHS=100; BATCH=64
print("Training Self-Attention Classifier ...")
for epoch in range(EPOCHS):
    model.train()
    idx  = torch.randperm(len(X_tr))[:BATCH]
    logits, _ = model(X_tr[idx])
    loss = crit(logits, y_tr[idx])
    opt.zero_grad(); loss.backward(); opt.step()
    if (epoch+1) % 25 == 0:
        model.eval()
        with torch.no_grad():
            test_logits, _ = model(X_te)
            acc = (test_logits.argmax(-1) == y_te).float().mean()
        print(f"  Epoch {epoch+1}  loss={loss.item():.4f}  test_acc={acc:.3f}")

model.eval()
with torch.no_grad():
    logits, attn = model(X_te[:8])
    preds = logits.argmax(-1)

print(f"\nSample predictions: {preds.tolist()}")
print(f"True labels:        {y_te[:8].tolist()}")

# ─── Visualize attention for 2 test samples ──────────────────────────────────
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for i, ax in enumerate(axes):
    w = attn[i].numpy()
    im = ax.imshow(w, cmap="viridis", vmin=0)
    ax.set_title(f"Sample {i} — Class={y_te[i].item()}  Pred={preds[i].item()}")
    ax.set_xlabel("Key position"); ax.set_ylabel("Query position")
    plt.colorbar(im, ax=ax)
plt.suptitle("Self-Attention Weights in Sequence Classifier", fontsize=10)
plt.tight_layout(); plt.savefig("self_attn_cls.png", dpi=85); plt.close()
print("Saved self_attn_cls.png")
