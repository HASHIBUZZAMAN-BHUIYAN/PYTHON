# Advanced Day 09 Mini-Project — PyTorch Multi-Class Digit Classifier
# ~300 MB RAM, ~3 min on CPU
# Scale up: increase EPOCHS or N_SAMPLES for better accuracy with more hardware.

import torch, torch.nn as nn, torch.optim as optim
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42); np.random.seed(42)

# ─── Config ──────────────────────────────────────────────────────────────────
N_SAMPLES = 1797   # all digits samples (~small enough for CPU)
EPOCHS     = 15
BATCH_SIZE = 32
LR         = 1e-3

# ─── Data ────────────────────────────────────────────────────────────────────
digits = load_digits()
sc = StandardScaler()
X = sc.fit_transform(digits.data).astype(np.float32)
y = digits.target

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
test_ds  = TensorDataset(torch.FloatTensor(X_te), torch.LongTensor(y_te))
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  BATCH_SIZE)

# ─── Model ───────────────────────────────────────────────────────────────────
class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,  64), nn.ReLU(),
            nn.Linear(64,   10),
        )
    def forward(self, x): return self.net(x)

model    = DigitNet()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion = nn.CrossEntropyLoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training {EPOCHS} epochs, batch {BATCH_SIZE} ...")

train_losses, train_accs, test_accs = [], [], []

for epoch in range(1, EPOCHS+1):
    model.train()
    t_loss, t_correct, t_total = 0., 0, 0
    for X_b, y_b in train_loader:
        optimizer.zero_grad()
        out  = model(X_b)
        loss = criterion(out, y_b)
        loss.backward(); optimizer.step()
        t_loss    += loss.item() * len(y_b)
        t_correct += (out.argmax(1) == y_b).sum().item()
        t_total   += len(y_b)
    scheduler.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            preds   = model(X_b).argmax(1)
            correct += (preds == y_b).sum().item()
            total   += len(y_b)

    tr_acc = t_correct / t_total
    te_acc = correct / total
    tr_loss = t_loss / t_total
    train_losses.append(tr_loss); train_accs.append(tr_acc); test_accs.append(te_acc)
    print(f"Epoch {epoch:2d}: train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}")

# ─── Final evaluation ────────────────────────────────────────────────────────
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for X_b, y_b in test_loader:
        all_preds.extend(model(X_b).argmax(1).numpy())
        all_true.extend(y_b.numpy())

print(f"\nFinal test accuracy: {sum(p==t for p,t in zip(all_preds,all_true))/len(all_true):.4f}")
print(classification_report(all_true, all_preds))

# ─── Visualise ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_losses, label="Train loss")
axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

axes[1].plot(train_accs, label="Train acc"); axes[1].plot(test_accs, label="Test acc")
axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()

plt.tight_layout(); plt.savefig("digit_pytorch.png", dpi=80)
print("\nSaved digit_pytorch.png")
