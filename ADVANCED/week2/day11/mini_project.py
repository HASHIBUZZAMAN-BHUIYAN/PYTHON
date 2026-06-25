# Advanced Day 11 Mini-Project — CNN on MNIST subset (CPU-friendly)
# ~300 MB RAM, ~5-8 min on CPU
# Uses only 5000 training samples to stay fast on 8GB/no-GPU machines.
# Scale up: set N_TRAIN=60000 for full MNIST (requires ~30 min on CPU).

import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# NOTE: ~300 MB RAM, ~5-8 min on CPU
# Scale up comment: set N_TRAIN=60000, EPOCHS=30 for full training (GPU recommended)

torch.manual_seed(42); np.random.seed(42)

N_TRAIN = 5000   # keep small for CPU; increase if you want better accuracy
EPOCHS  = 10
BATCH   = 64
LR      = 1e-3

print(f"Loading MNIST subset ({N_TRAIN} samples) ...")
try:
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int)
    idx = np.random.choice(len(X), N_TRAIN + 1000, replace=False)
    X, y = X[idx], y[idx]
    print(f"Loaded MNIST: using {N_TRAIN} train + 1000 test samples")
except Exception as e:
    print(f"MNIST download failed ({e}), using sklearn digits instead ...")
    from sklearn.datasets import load_digits
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target
    # Upsample to 28x28 for consistency
    import torch.nn.functional as F2
    X_t = torch.FloatTensor(X).view(-1, 1, 8, 8)
    X = F2.interpolate(X_t, 28, mode="bilinear", align_corners=False).view(-1, 784).numpy()
    N_TRAIN = min(N_TRAIN, len(X) - 200)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=min(1000, len(X)//5), stratify=y, random_state=42)
X_tr, X_te = X_tr[:N_TRAIN], X_te[:1000]
y_tr, y_te = y_tr[:N_TRAIN], y_te[:1000]

X_tr_t = torch.FloatTensor(X_tr).view(-1, 1, 28, 28)
X_te_t = torch.FloatTensor(X_te).view(-1, 1, 28, 28)
y_tr_t, y_te_t = torch.LongTensor(y_tr), torch.LongTensor(y_te)
train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), BATCH, shuffle=True)

# ─── CNN Architecture ────────────────────────────────────────────────────────
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 28×28 → Conv(1→16,3,p=1) → 28×28 → Pool → 14×14
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2)
        )
        # 14×14 → Conv(16→32,3,p=1) → 14×14 → Pool → 7×7
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2)
        )
        # 7×7 → Conv(32→64,3,p=1) → 7×7
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),         # → (batch, 64, 1, 1)
            nn.Flatten(),                    # → (batch, 64)
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        return self.classifier(x)

model = MNIST_CNN()
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

optimizer = optim.Adam(model.parameters(), LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
criterion = nn.CrossEntropyLoss()

train_losses, test_accs = [], []
for epoch in range(1, EPOCHS+1):
    model.train(); tl, tc, tt = 0., 0, 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward(); optimizer.step()
        tl += loss.item()*len(yb); tc += (out.argmax(1)==yb).sum().item(); tt += len(yb)
    scheduler.step()

    model.eval()
    with torch.no_grad():
        te_preds = model(X_te_t).argmax(1)
        te_acc   = (te_preds == y_te_t).float().mean().item()
    train_losses.append(tl/tt); test_accs.append(te_acc)
    print(f"Ep {epoch:2d}: loss={tl/tt:.4f} train_acc={tc/tt:.4f} test_acc={te_acc:.4f}")

print("\n" + classification_report(y_te, te_preds.numpy()))

# ─── Visualise ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(train_losses, marker="o"); axes[0].set_title("Train Loss")
axes[1].plot(test_accs, marker="o", color="green"); axes[1].set_title("Test Accuracy")
plt.tight_layout(); plt.savefig("mnist_cnn.png", dpi=80)
print("Saved mnist_cnn.png")
plt.show()
