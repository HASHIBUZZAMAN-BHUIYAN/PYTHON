# DL Reference — PyTorch Training Loop Template
# CPU-friendly. Swap in your own model/dataset.
# ~60 MB RAM, <5s on CPU (tiny demo data)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# ─── 1. MODEL ─────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_features, hidden, out_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden//2),   nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden//2, out_classes),
        )
    def forward(self, x): return self.net(x)

# ─── 2. DATA ─────────────────────────────────────────────────────────────────
# Replace with your own dataset:
# dataset = MyDataset(...)
# train_dl = DataLoader(dataset, batch_size=32, shuffle=True)

np.random.seed(42); torch.manual_seed(42)
N_TRAIN, N_VAL = 800, 200
X = torch.randn(N_TRAIN + N_VAL, 20)
y = (X[:, :5].sum(1) > 0).long()
train_dl = DataLoader(TensorDataset(X[:N_TRAIN], y[:N_TRAIN]), batch_size=32, shuffle=True)
val_dl   = DataLoader(TensorDataset(X[N_TRAIN:], y[N_TRAIN:]), batch_size=64)

# ─── 3. TRAINING CONFIG ──────────────────────────────────────────────────────
model      = MLP(20, 64, 2)
criterion  = nn.CrossEntropyLoss()
optimizer  = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
N_EPOCHS   = 15
PATIENCE   = 4        # early stopping

device = torch.device("cpu")
model  = model.to(device)

# ─── 4. TRAINING LOOP ────────────────────────────────────────────────────────
history = {"train_loss":[], "val_loss":[], "val_acc":[]}
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(1, N_EPOCHS + 1):
    # ── Train ──
    model.train()
    train_loss = 0.
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(train_dl.dataset)

    # ── Validate ──
    model.eval()
    val_loss, correct = 0., 0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            val_loss += criterion(logits, yb).item() * len(xb)
            correct  += (logits.argmax(1) == yb).sum().item()
    val_loss /= len(val_dl.dataset)
    val_acc   = correct / len(val_dl.dataset)

    scheduler.step()
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch:>3}/{N_EPOCHS}  "
          f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

    # ── Early stopping ──
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

# ─── 5. PLOT ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history["train_loss"], label="Train loss")
axes[0].plot(history["val_loss"],   label="Val loss")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(history["val_acc"], color="green")
axes[1].set_title("Val Accuracy"); axes[1].set_ylim(0,1); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("pytorch_training.png", dpi=80)
print("Saved pytorch_training.png")

# ─── 6. LOAD BEST & EVALUATE ─────────────────────────────────────────────────
model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
model.eval()
with torch.no_grad():
    preds = model(X[N_TRAIN:].to(device)).argmax(1).cpu()
final_acc = (preds == y[N_TRAIN:]).float().mean()
print(f"Final val accuracy (best checkpoint): {final_acc:.4f}")
