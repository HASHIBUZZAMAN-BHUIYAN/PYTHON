"""
Day 29 Project 3: Sequence Pattern Classifier
==============================================
Classify binary sequences by pattern:
  - Class 0: more 1s in the FIRST half  (e.g., [1,1,0,1,0 | 0,0,0,0,1])
  - Class 1: more 1s in the SECOND half (e.g., [0,0,0,1,0 | 1,1,1,0,1])

200 synthetic sequences of length 10.
Architecture: nn.RNN, hidden_size=16
Training: 15 epochs

~50 MB RAM, ~5s on CPU
"""

import torch
import torch.nn as nn
import numpy as np
import time

torch.manual_seed(42)
np.random.seed(42)

# ── Generate Data ─────────────────────────────────────────────────────────────
N        = 200
SEQ_LEN  = 10
HALF     = SEQ_LEN // 2

def make_dataset(n=200):
    X, y = [], []
    for _ in range(n):
        # coin flip to pick class
        cls = np.random.randint(0, 2)
        seq = np.zeros(SEQ_LEN, dtype=np.float32)

        if cls == 0:
            # more 1s in first half
            first_half  = np.random.choice([0.0, 1.0], size=HALF,
                                            p=[0.25, 0.75])
            second_half = np.random.choice([0.0, 1.0], size=HALF,
                                            p=[0.75, 0.25])
        else:
            # more 1s in second half
            first_half  = np.random.choice([0.0, 1.0], size=HALF,
                                            p=[0.75, 0.25])
            second_half = np.random.choice([0.0, 1.0], size=HALF,
                                            p=[0.25, 0.75])

        seq[:HALF]  = first_half
        seq[HALF:]  = second_half
        X.append(seq)
        y.append(cls)

    return np.array(X), np.array(y)

X_np, y_np = make_dataset(N)

# Verify the distribution makes sense
c0_first_half_mean  = X_np[y_np==0, :HALF].mean()
c0_second_half_mean = X_np[y_np==0, HALF:].mean()
c1_first_half_mean  = X_np[y_np==1, :HALF].mean()
c1_second_half_mean = X_np[y_np==1, HALF:].mean()
print(f"Class 0 — first half 1s: {c0_first_half_mean:.2f}, second half: {c0_second_half_mean:.2f}")
print(f"Class 1 — first half 1s: {c1_first_half_mean:.2f}, second half: {c1_second_half_mean:.2f}")
print(f"Class balance: {(y_np==0).sum()} class-0, {(y_np==1).sum()} class-1\n")

# Train / test split (80/20)
split   = int(0.8 * N)
X_train = torch.tensor(X_np[:split]).unsqueeze(-1)   # [160, 10, 1]
y_train = torch.tensor(y_np[:split], dtype=torch.long)
X_test  = torch.tensor(X_np[split:]).unsqueeze(-1)   # [40, 10, 1]
y_test  = torch.tensor(y_np[split:], dtype=torch.long)

# ── Model ─────────────────────────────────────────────────────────────────────
class SeqClassifier(nn.Module):
    def __init__(self, hidden_size=16, n_classes=2):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.fc(h_n.squeeze(0))

model     = SeqClassifier(hidden_size=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params}")

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS     = 15
BATCH_SIZE = 32

print("Training...")
start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    perm       = torch.randperm(len(X_train))
    total_loss = 0.0
    n_batches  = 0

    for i in range(0, len(X_train), BATCH_SIZE):
        idx    = perm[i : i+BATCH_SIZE]
        xb, yb = X_train[idx], y_train[idx]
        logits = model(xb)
        loss   = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    if epoch % 5 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            preds_tr  = model(X_train).argmax(dim=1)
            train_acc = (preds_tr == y_train).float().mean().item()
        print(f"  Epoch {epoch:2d} | Loss: {total_loss/n_batches:.4f} | TrainAcc: {train_acc:.2%}")

elapsed = time.time() - start
print(f"\nTraining complete in {elapsed:.1f}s")

# ── Evaluation ────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    logits_test = model(X_test)
    preds_test  = logits_test.argmax(dim=1)
    test_acc    = (preds_test == y_test).float().mean().item()

print(f"\nTest Accuracy: {test_acc:.2%} on {len(X_test)} samples")

# Per-class accuracy
for cls in [0, 1]:
    mask    = y_test == cls
    cls_acc = (preds_test[mask] == y_test[mask]).float().mean().item()
    print(f"  Class {cls} accuracy: {cls_acc:.2%}  ({mask.sum()} samples)")

# Baseline: predict most common class always
baseline = max((y_test==0).float().mean(), (y_test==1).float().mean()).item()
print(f"\nMajority-class baseline: {baseline:.2%}")
print(f"RNN improvement over baseline: +{(test_acc - baseline)*100:.1f}%")
