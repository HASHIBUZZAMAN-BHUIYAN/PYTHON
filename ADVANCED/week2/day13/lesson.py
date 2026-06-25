# Advanced Day 13 — Transfer Learning (Lightweight, CPU-friendly)
# NOTE: heavier lesson — close other apps before running
# ~500 MB RAM, ~5-10 min on CPU
# We freeze the backbone and only train the classification head.
# Scale up: unfreeze more layers for better accuracy with GPU.

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms

torch.manual_seed(42); np.random.seed(42)

# ─── 1. WHAT IS TRANSFER LEARNING? ────────────────────────────────────────────
print("""
=== Transfer Learning Concept ===

Pretrained CNN (e.g. MobileNetV2, trained on ImageNet):
  - Already learned to detect edges, textures, shapes, objects
  - Feature extractor layers are FROZEN (no gradient update)
  - Only the classification HEAD is retrained on our data

Why MobileNetV2 for this CPU demo?
  - Only ~3.4 MB model size
  - Designed for mobile/edge devices — fast even on CPU
  - Still achieves ~71% top-1 accuracy on ImageNet

Two strategies:
  1. Feature extraction: freeze ALL conv layers, train only final Linear
  2. Fine-tuning: unfreeze last few layers and train with small LR
  (we use strategy 1 here — much faster on CPU)
""")

# ─── 2. SYNTHETIC IMAGE DATASET ──────────────────────────────────────────────
print("=== 2. Generating Synthetic Shapes Dataset ===")

def make_shape_image(shape_type, size=32):
    """Generate a synthetic 32x32 image with a simple shape."""
    img = np.ones((size, size, 3), dtype=np.float32) * 0.9  # light background
    center = size // 2
    radius = size // 4

    if shape_type == 0:   # Circle
        for y in range(size):
            for x in range(size):
                if (x-center)**2 + (y-center)**2 <= radius**2:
                    img[y, x] = [0.2, 0.4, 0.8]
    elif shape_type == 1: # Square
        img[center-radius:center+radius, center-radius:center+radius] = [0.8, 0.2, 0.2]
    elif shape_type == 2: # Triangle (approximate)
        for y in range(size):
            half_w = int((y - (center-radius)) * radius / radius) if y > center-radius else 0
            if center-radius <= y <= center+radius and center-half_w <= center+half_w:
                img[y, max(0,center-half_w):min(size,center+half_w)] = [0.2, 0.8, 0.2]
    # Add noise
    img += np.random.normal(0, 0.05, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)

# Generate 300 samples per class (3 classes)
N_PER_CLASS = 300
X_list, y_list = [], []
for class_id in range(3):
    for _ in range(N_PER_CLASS):
        X_list.append(make_shape_image(class_id))
        y_list.append(class_id)

X_all = np.array(X_list); y_all = np.array(y_list)
print(f"Dataset: {X_all.shape}  (N, H, W, C)")

# Shuffle
idx = np.random.permutation(len(X_all))
X_all, y_all = X_all[idx], y_all[idx]
split = int(0.8 * len(X_all))
X_tr, X_te = X_all[:split], X_all[split:]
y_tr, y_te = y_all[:split], y_all[split:]

# Convert to PyTorch tensors (N, C, H, W)
to_tensor = lambda X: torch.FloatTensor(X).permute(0, 3, 1, 2)
X_tr_t, X_te_t = to_tensor(X_tr), to_tensor(X_te)
y_tr_t, y_te_t = torch.LongTensor(y_tr), torch.LongTensor(y_te)

# Normalize to ImageNet stats (required for pretrained models)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
X_tr_t = (X_tr_t - mean) / std
X_te_t = (X_te_t - mean) / std

# Resize from 32×32 to 96×96 (MobileNetV2 minimum is 96)
X_tr_t = F.interpolate(X_tr_t, 96)
X_te_t = F.interpolate(X_te_t, 96)

# ─── 3. FEATURE EXTRACTION WITH MOBILENETV2 ──────────────────────────────────
print("\n=== 3. Feature Extraction with MobileNetV2 ===")
backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Freeze ALL backbone parameters
for param in backbone.parameters():
    param.requires_grad = False

# Replace classifier head for 3 classes
n_features = backbone.classifier[1].in_features
backbone.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(n_features, 3)
)

trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
total     = sum(p.numel() for p in backbone.parameters())
print(f"Total params: {total:,}  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

# ─── 4. TRAINING THE HEAD ────────────────────────────────────────────────────
print("\n=== 4. Training Classification Head ===")
# Only optimize classifier params (backbone is frozen)
optimizer = optim.Adam(backbone.classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
loader    = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)

EPOCHS = 10
losses, te_accs = [], []

for epoch in range(1, EPOCHS+1):
    backbone.train()
    epoch_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        out  = backbone(xb)
        loss = criterion(out, yb)
        loss.backward(); optimizer.step()
        epoch_loss += loss.item()

    backbone.eval()
    with torch.no_grad():
        preds = backbone(X_te_t).argmax(1)
        acc   = (preds == y_te_t).float().mean().item()
    losses.append(epoch_loss / len(loader))
    te_accs.append(acc)
    print(f"Epoch {epoch:2d}: loss={losses[-1]:.4f}  test_acc={acc:.4f}")

# ─── 5. COMPARE TO TRAINING FROM SCRATCH ─────────────────────────────────────
print("\n=== 5. Comparison: Pretrained vs Scratch ===")
scratch = nn.Sequential(
    nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
    nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
    nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),
    nn.Flatten(), nn.Linear(64,3)
)
opt_s = optim.Adam(scratch.parameters(), 1e-3)
sc_accs = []
for ep in range(EPOCHS):
    scratch.train()
    for xb,yb in loader:
        opt_s.zero_grad(); l=criterion(scratch(xb),yb); l.backward(); opt_s.step()
    scratch.eval()
    with torch.no_grad():
        sc_accs.append((scratch(X_te_t).argmax(1)==y_te_t).float().mean().item())
print(f"Pretrained final: {te_accs[-1]:.4f}  Scratch final: {sc_accs[-1]:.4f}")

# ─── 6. VISUALISE ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# Sample images
axes[0].set_title("Sample Shapes")
for i, (name, class_id) in enumerate([("Circle",0),("Square",1),("Triangle",2)]):
    img = make_shape_image(class_id)
    axes[0].imshow(img, extent=[i,i+1,0,1]); axes[0].text(i+0.5, -0.15, name, ha="center")
axes[0].set_xlim(0,3); axes[0].set_ylim(-0.2,1); axes[0].axis("off")

axes[1].plot(te_accs, marker="o", label="Pretrained (frozen)")
axes[1].plot(sc_accs, marker="s", linestyle="--", label="From scratch")
axes[1].set_title("Test Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()

axes[2].plot(losses, color="tomato", marker="o")
axes[2].set_title("Training Loss (Pretrained)"); axes[2].set_xlabel("Epoch")

plt.tight_layout(); plt.savefig("transfer_learning.png", dpi=80)
print("\nSaved transfer_learning.png")
