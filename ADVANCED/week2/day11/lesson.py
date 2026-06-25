# Advanced Day 11 — CNN Fundamentals
# ~300 MB RAM, ~5-8 min on CPU (small dataset + small model)
# Scale up: increase N_TRAIN or EPOCHS for better results

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42); torch.manual_seed(42)

# ─── 1. CONVOLUTION EXPLAINED ────────────────────────────────────────────────
print("=== 1. What is a Convolution? ===")
print("""
A convolution slides a small filter (kernel) over the input.
At each position it computes a dot product: output[i,j] = sum(input[i:i+k, j:j+k] * kernel)
This detects local patterns (edges, corners, textures).

Input(H×W) + Filter(k×k) → Output((H-k+1)×(W-k+1))
With padding='same' → output same size as input.
""")

# Manual 2D convolution demonstration
def conv2d_manual(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    out = np.zeros((h-kh+1, w-kw+1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i,j] = (image[i:i+kh, j:j+kw] * kernel).sum()
    return out

# Synthetic image
image = np.zeros((8,8)); image[2:6, 2:6] = 1
print("Image (8×8):\n", image.astype(int))

# Edge detection kernel (Sobel-like)
sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=float)
feature_map = conv2d_manual(image, sobel_x)
print("Sobel-X feature map:\n", feature_map.astype(int))

# ─── 2. POOLING ──────────────────────────────────────────────────────────────
print("\n=== 2. Pooling ===")
print("""
Max pooling takes the maximum value in each non-overlapping region.
Pool(2×2) on 4×4 → 2×2: reduces spatial size, keeps dominant features.
Average pooling takes the mean instead.
""")
x_pool = np.array([[1,3,2,4],[5,6,7,8],[3,2,1,0],[9,8,7,6]])
print("Input:\n", x_pool)
pool_out = np.zeros((2,2))
for i in range(2):
    for j in range(2):
        pool_out[i,j] = x_pool[i*2:i*2+2, j*2:j*2+2].max()
print("MaxPool(2x2):\n", pool_out)

# ─── 3. CNN ARCHITECTURE OVERVIEW ────────────────────────────────────────────
print("\n=== 3. Typical CNN Architecture ===")
print("""
Input → [Conv → ReLU → Pool] × N → Flatten → [Dense] × M → Output

Conv extracts spatial features.
Pool reduces size (downsampling).
Flatten converts 2D feature maps to 1D vector for dense layers.
Output: softmax for classification.
""")

# ─── 4. BUILD A CNN WITH PYTORCH ─────────────────────────────────────────────
print("=== 4. Building CNN in PyTorch ===")

class SmallCNN(nn.Module):
    """2-layer CNN for 8×8 grayscale images (digits dataset)."""
    def __init__(self, n_classes=10):
        super().__init__()
        # Conv block 1: 1 channel → 8 channels, 3×3 kernel, pad=1 keeps size
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)   # 8×8 → 4×4
        # Conv block 2: 8 → 16 channels, 3×3 kernel
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # After pool again: 4×4 → 2×2
        self.fc1   = nn.Linear(16 * 2 * 2, 64)
        self.fc2   = nn.Linear(64, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: (batch, 1, 8, 8)
        x = self.pool(F.relu(self.conv1(x)))   # → (batch, 8, 4, 4)
        x = self.pool(F.relu(self.conv2(x)))   # → (batch, 16, 2, 2)
        x = x.view(x.size(0), -1)             # flatten → (batch, 64)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

model = SmallCNN()
print(model)
# Count params
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

# Test forward pass
dummy = torch.zeros(4, 1, 8, 8)
print("Output shape:", model(dummy).shape)   # (4, 10)

# ─── 5. TRAIN ON DIGITS (8×8 images) ─────────────────────────────────────────
print("\n=== 5. Training CNN on Digits ===")
digits = load_digits()
X = digits.data.astype(np.float32) / 16.0    # normalize 0..1
y = digits.target

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Reshape to (N, 1, 8, 8) for CNN
X_tr_t = torch.FloatTensor(X_tr).view(-1, 1, 8, 8)
X_te_t = torch.FloatTensor(X_te).view(-1, 1, 8, 8)
y_tr_t = torch.LongTensor(y_tr)
y_te_t = torch.LongTensor(y_te)

loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_losses, test_accs = [], []
for epoch in range(1, 16):
    model.train(); total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward(); optimizer.step()
        total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        preds = model(X_te_t).argmax(1)
        acc   = (preds == y_te_t).float().mean().item()
    avg_loss = total_loss / len(loader)
    train_losses.append(avg_loss); test_accs.append(acc)
    print(f"Epoch {epoch:2d}: loss={avg_loss:.4f}  test_acc={acc:.4f}")

# ─── 6. VISUALISE FEATURE MAPS ────────────────────────────────────────────────
model.eval()
sample = X_te_t[0:1]  # one test image
with torch.no_grad():
    feat_maps = F.relu(model.conv1(sample))  # (1, 8, 4, 4)... wait, conv1 before pool

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("CNN First-Layer Feature Maps (sample digit)")
axes[0,0].imshow(sample[0,0].numpy(), cmap="gray")
axes[0,0].set_title("Input"); axes[0,0].axis("off")
for i in range(min(8, feat_maps.shape[1])):
    ax = axes[(i+1)//5, (i+1)%5]
    ax.imshow(feat_maps[0,i].numpy(), cmap="viridis")
    ax.set_title(f"Filter {i}"); ax.axis("off")
for ax in axes.flat:
    ax.axis("off")
plt.tight_layout(); plt.savefig("cnn_features.png", dpi=80)
print("\nSaved cnn_features.png")
