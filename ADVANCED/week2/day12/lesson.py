# Advanced Day 12 — CNN Training & Tuning
# ~400 MB RAM, ~8-12 min on CPU

import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(42); np.random.seed(42)

digits = load_digits()
X = digits.data.astype(np.float32) / 16.0
y = digits.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def to_tensor(X, y):
    return torch.FloatTensor(X).view(-1, 1, 8, 8), torch.LongTensor(y)

X_tr_t, y_tr_t = to_tensor(X_tr, y_tr)
X_te_t, y_te_t = to_tensor(X_te, y_te)

# ─── 1. DATA AUGMENTATION ────────────────────────────────────────────────────
print("=== 1. Data Augmentation ===")
print("""
Augmentation artificially increases training data variety.
On 8×8 digits we can:
  - Add gaussian noise
  - Random horizontal/vertical shifts (±1 pixel)
  - Slight brightness perturbation

NOTE: For ImageNet-scale images you would use torchvision.transforms:
  RandomHorizontalFlip, RandomCrop, ColorJitter, RandomRotation, Normalize
""")

def augment_batch(xb):
    """Simple CPU augmentation for tiny 8×8 images."""
    # Random noise
    noise = torch.randn_like(xb) * 0.05
    xb = torch.clamp(xb + noise, 0, 1)
    # Random brightness
    factor = 0.9 + torch.rand(1).item() * 0.2
    xb = torch.clamp(xb * factor, 0, 1)
    return xb

# ─── 2. BASELINE CNN ─────────────────────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self, use_bn=True, dropout=0.3):
        super().__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(16) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(32) if use_bn else nn.Identity()
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 2 * 2, 64)
        self.fc2   = nn.Linear(64, 10)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        return self.fc2(self.drop(F.relu(self.fc1(x))))

def train_one(model, n_epochs, lr, use_aug=False, label=""):
    opt  = optim.Adam(model.parameters(), lr)
    crit = nn.CrossEntropyLoss()
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    tr_accs, te_accs = [], []
    for ep in range(n_epochs):
        model.train()
        loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), 32, shuffle=True)
        for xb, yb in loader:
            if use_aug: xb = augment_batch(xb)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            tr_acc = (model(X_tr_t).argmax(1)==y_tr_t).float().mean().item()
            te_acc = (model(X_te_t).argmax(1)==y_te_t).float().mean().item()
        tr_accs.append(tr_acc); te_accs.append(te_acc)
    print(f"{label}: final test_acc={te_accs[-1]:.4f}")
    return tr_accs, te_accs

n_epochs = 20
configs = [
    ("No BN, no aug",   CNN(use_bn=False, dropout=0),   False),
    ("BN + Dropout",    CNN(use_bn=True,  dropout=0.3), False),
    ("BN + Dropout + Aug", CNN(use_bn=True, dropout=0.3), True),
]
results = {}
for label, model, aug in configs:
    tr, te = train_one(model, n_epochs, 1e-3, use_aug=aug, label=label)
    results[label] = (tr, te)

# ─── 3. LEARNING RATE COMPARISON ────────────────────────────────────────────
print("\n=== 3. Learning Rate Comparison ===")
lr_results = {}
for lr in [1e-2, 1e-3, 1e-4]:
    m = CNN(); tr, te = train_one(m, 15, lr, label=f"lr={lr}")
    lr_results[f"lr={lr}"] = te

# ─── 4. VISUALISE ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for label, (tr, te) in results.items():
    axes[0].plot(te, label=label)
axes[0].set_title("Test Accuracy: Tuning Techniques")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Test Acc")
axes[0].legend(fontsize=8)

for label, te in lr_results.items():
    axes[1].plot(te, label=label)
axes[1].set_title("Test Accuracy: Learning Rate")
axes[1].set_xlabel("Epoch"); axes[1].legend()

plt.tight_layout(); plt.savefig("cnn_tuning.png", dpi=80)
print("\nSaved cnn_tuning.png")
