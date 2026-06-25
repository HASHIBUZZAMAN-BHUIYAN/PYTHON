"""
human_text_understanding.py
============================
What it does:
    Creates 200 synthetic IT support messages labeled urgent/not-urgent,
    trains a small embedding + feed-forward classifier in PyTorch, and
    analyses which vocabulary tokens carry the most signal by examining
    their embedding L2 norms.

What it teaches:
    - Building a text pipeline from scratch (tokenise, vocab, pad)
    - nn.Embedding for trainable word representations
    - Mean-pooling an embedding sequence before classification
    - Interpreting trained embeddings via token norms

RAM estimate : ~20 MB
Time estimate: ~10 seconds on CPU (Ryzen 7)
Real vs simulated: ALL messages are synthetically constructed from
    template phrases. No real customer-support corpus is used.
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib_available = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib_available = False

os.makedirs("DL/outputs", exist_ok=True)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ── 1. Synthetic corpus ───────────────────────────────────────────────────────
URGENT_TEMPLATES = [
    "system is completely down",
    "critical error in production",
    "cannot access the application",
    "severe data loss detected",
    "database is not responding",
    "service outage affecting all users",
    "urgent fix required immediately",
    "production server crashed",
    "payment system failing",
    "security breach detected",
    "application is completely broken",
    "data corruption in main database",
    "all users are locked out",
    "critical system failure occurred",
    "emergency maintenance required now",
]

NON_URGENT_TEMPLATES = [
    "would love a dark mode option",
    "minor cosmetic issue in the dashboard",
    "feature request for export to csv",
    "nice to have better font rendering",
    "suggestion to improve the colour scheme",
    "small typo on the about page",
    "could you add a tooltip here",
    "low priority improvement for later",
    "the button could be slightly bigger",
    "requesting a new report template",
    "it would be nice to have autocomplete",
    "minor formatting inconsistency noticed",
    "suggestion for improved documentation",
    "enhancement request not blocking anyone",
    "cosmetic polish for the sidebar",
]

def augment(template, rng_local):
    """Slightly vary a template by shuffling or dropping a word."""
    words = template.split()
    if len(words) > 3 and rng_local.random() < 0.4:
        i = rng_local.randint(1, len(words) - 1)
        words.pop(i)
    return " ".join(words)

rng_local = random.Random(99)
urgent_msgs     = [augment(random.choice(URGENT_TEMPLATES),     rng_local) for _ in range(100)]
non_urgent_msgs = [augment(random.choice(NON_URGENT_TEMPLATES), rng_local) for _ in range(100)]

messages = urgent_msgs + non_urgent_msgs
labels   = [1] * 100 + [0] * 100

# Shuffle together
combined = list(zip(messages, labels))
random.shuffle(combined)
messages, labels = zip(*combined)
messages, labels = list(messages), list(labels)

print(f"[OK] Corpus: {len(messages)} messages  ({sum(labels)} urgent, "
      f"{len(labels)-sum(labels)} not-urgent)")
print(f"  Sample urgent    : {messages[labels.index(1)]}")
print(f"  Sample not-urgent: {messages[labels.index(0)]}")

# ── 2. Tokenise and build vocab ───────────────────────────────────────────────
MAX_LEN = 15
PAD_IDX = 0
UNK_IDX = 1

from collections import Counter
all_tokens = [tok for msg in messages for tok in msg.lower().split()]
freq = Counter(all_tokens)
# Build vocab: 0=PAD, 1=UNK, then by frequency
vocab = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
for tok, _ in freq.most_common():
    vocab[tok] = len(vocab)

VOCAB_SIZE = len(vocab)
print(f"[OK] Vocabulary size: {VOCAB_SIZE}")

def encode(msg, max_len):
    toks = [vocab.get(t, UNK_IDX) for t in msg.lower().split()[:max_len]]
    toks += [PAD_IDX] * (max_len - len(toks))
    return toks

X_enc = np.array([encode(m, MAX_LEN) for m in messages], dtype=np.int64)
y_arr = np.array(labels, dtype=np.float32)

# Train / test split (80/20)
split = int(0.8 * len(X_enc))
X_tr, X_te = X_enc[:split], X_enc[split:]
y_tr, y_te = y_arr[:split], y_arr[split:]

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
    batch_size=32, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
    batch_size=32, shuffle=False
)

# ── 3. Model ──────────────────────────────────────────────────────────────────
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.fc1   = nn.Linear(embed_dim, 32)
        self.fc2   = nn.Linear(32, 1)
        self.drop  = nn.Dropout(0.3)

    def forward(self, x):
        # x: (batch, seq_len)
        mask    = (x != 0).float().unsqueeze(-1)     # (B, L, 1)
        emb     = self.embed(x)                       # (B, L, E)
        pooled  = (emb * mask).sum(1) / (mask.sum(1) + 1e-8)   # mean-pool
        pooled  = self.drop(pooled)
        h       = torch.relu(self.fc1(pooled))
        return self.fc2(h).squeeze(-1)

model     = TextClassifier(VOCAB_SIZE, embed_dim=16)
total_p   = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
criterion = nn.BCEWithLogitsLoss()
print(f"[OK] Model params: {total_p:,}")

# ── 4. Training ────────────────────────────────────────────────────────────────
EPOCHS = 15
train_losses = []
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(xb)
        preds    = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == yb).sum().item()
        total   += len(yb)
    avg_loss = running_loss / total
    train_losses.append(avg_loss)
    if epoch == 1 or epoch % 3 == 0:
        print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  "
              f"train_acc={correct/total:.3f}")

elapsed = time.time() - t0
print(f"[OK] Training done in {elapsed:.1f}s")

# ── 5. Test accuracy ───────────────────────────────────────────────────────────
model.eval()
all_preds, all_labels_eval = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb)
        preds  = (torch.sigmoid(logits) > 0.5).float()
        all_preds.extend(preds.numpy())
        all_labels_eval.extend(yb.numpy())

all_preds       = np.array(all_preds)
all_labels_eval = np.array(all_labels_eval)
test_acc = (all_preds == all_labels_eval).mean()
print(f"\n[OK] Test accuracy: {test_acc:.3f}")

# Sample predictions on a few held-out messages
print("\nSample predictions:")
model.eval()
with torch.no_grad():
    sample_msgs = [
        "critical system is down now",
        "nice to have a dark theme",
        "database error in production",
        "feature request low priority",
    ]
    for msg in sample_msgs:
        enc = torch.tensor([encode(msg, MAX_LEN)], dtype=torch.long)
        logit = model(enc).item()
        prob  = 1 / (1 + np.exp(-logit))
        label = "URGENT" if prob > 0.5 else "not-urgent"
        print(f"  [{label:10s}] (p={prob:.2f}) - {msg}")

# ── 6. Most important words by embedding norm ─────────────────────────────────
embed_weights = model.embed.weight.detach().numpy()   # (vocab_size, embed_dim)
norms = np.linalg.norm(embed_weights, axis=1)          # L2 norm per token

# Map back from index to word
idx2word = {v: k for k, v in vocab.items()}
word_norms = [(idx2word.get(i, "?"), norms[i]) for i in range(len(norms))
              if i not in (PAD_IDX,)]
word_norms.sort(key=lambda x: -x[1])

print("\nTop 15 words by embedding L2 norm (signal strength):")
for word, norm_val in word_norms[:15]:
    bar = int(norm_val * 4)
    print(f"  {word:20s} {norm_val:.3f}  {'|' * min(bar, 40)}")

# ── 7. Plots ──────────────────────────────────────────────────────────────────
if matplotlib_available:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(range(1, EPOCHS + 1), train_losses, marker="o", markersize=3,
                 color="steelblue")
    axes[0].set_title("Training Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].grid(True, alpha=0.3)

    # Top words by norm
    top_words = [w for w, _ in word_norms[:10]]
    top_norms = [n for _, n in word_norms[:10]]
    axes[1].barh(top_words[::-1], top_norms[::-1], color="salmon")
    axes[1].set_title("Top 10 Words by Embedding Norm\n(proxy for importance)")
    axes[1].set_xlabel("L2 Norm")

    plt.tight_layout()
    plt.savefig("DL/outputs/text_understanding_loss.png", dpi=100)
    plt.close()
    print("\n[OK] Plot saved to DL/outputs/text_understanding_loss.png")

print("\n[DONE] human_text_understanding.py complete")
