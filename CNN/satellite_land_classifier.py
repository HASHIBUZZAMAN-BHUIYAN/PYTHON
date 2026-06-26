"""
Satellite Land-Use Classifier
================================
What it does:
  Generates synthetic satellite-style patch images (64x64 px) for 4
  land-use categories -- water, forest, urban, farmland -- then trains
  a small CNN classifier on them. After training, builds a 6x6 "map" grid,
  runs the CNN on each cell, and overlays the predicted label so the grid
  looks like a land-use map.

  Real-world mirror: satellite imagery CNNs (e.g. EuroSAT dataset,
  Sentinel-2 patches) do exactly this -- patch-based land-use/land-cover
  classification from aerial or satellite tiles. Real models use actual
  multispectral imagery and transfer learning from ImageNet. Here we
  generate plausible colour+texture patches to keep it fully offline.

What it teaches:
  - Patch-based image classification: why land-use analysis tiles the
    image into NxN patches and classifies each independently
  - How colour and texture cues let a CNN distinguish land-cover types
  - How to visualise CNN predictions as a spatial prediction map
  - Class activation / confidence: softmax probability as a certainty bar

How to run:
  python CNN\satellite_land_classifier.py   (from PYTHON\ folder)

Estimated RAM: ~300MB | Time: ~60s (200 patches x 5 epochs, CPU)
Model: 3-conv-layer CNN, ~150k params, fully synthetic data, no download.
Output: CNN\outputs\satellite_land_map.png
"""

import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

os.makedirs("CNN/outputs", exist_ok=True)
rng = np.random.RandomState(42)
torch.manual_seed(42)

# ─── SYNTHETIC PATCH GENERATOR ────────────────────────────────────────────────

SIZE   = 64   # patch size
LABELS = ["water", "forest", "urban", "farmland"]

def gen_water(n):
    """Blue-dominant with horizontal ripple texture."""
    imgs = []
    for _ in range(n):
        img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        img[:, :, 2] = rng.randint(140, 200, (SIZE, SIZE))  # blue
        img[:, :, 1] = rng.randint(60,  120, (SIZE, SIZE))  # some green
        img[:, :, 0] = rng.randint(10,  50,  (SIZE, SIZE))  # little red
        # horizontal wave texture
        for row in range(0, SIZE, 4):
            img[row:row+2, :, 2] = np.clip(img[row:row+2, :, 2] + 30, 0, 255)
        imgs.append(img)
    return imgs

def gen_forest(n):
    """Dark green with varied canopy texture."""
    imgs = []
    for _ in range(n):
        img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        img[:, :, 1] = rng.randint(80,  160, (SIZE, SIZE))  # green channel dominant
        img[:, :, 0] = rng.randint(20,  60,  (SIZE, SIZE))
        img[:, :, 2] = rng.randint(10,  50,  (SIZE, SIZE))
        # random dark circles to simulate canopy shadows
        for _ in range(rng.randint(5, 15)):
            cx, cy, r = rng.randint(5, SIZE-5), rng.randint(5, SIZE-5), rng.randint(4, 10)
            yy, xx = np.ogrid[:SIZE, :SIZE]
            mask = (xx - cx)**2 + (yy - cy)**2 < r**2
            img[mask, 1] = np.clip(img[mask, 1].astype(int) - 30, 0, 255)
        imgs.append(img)
    return imgs

def gen_urban(n):
    """Grey dominant with rectangular grid structures."""
    imgs = []
    for _ in range(n):
        base = rng.randint(90, 160)
        img  = np.full((SIZE, SIZE, 3), base, dtype=np.uint8)
        img += rng.randint(-20, 20, (SIZE, SIZE, 3)).astype(np.int16).clip(0, 255).astype(np.uint8)
        # grid of building blocks
        for r in range(0, SIZE, 12):
            for c in range(0, SIZE, 12):
                h, w = rng.randint(6, 11), rng.randint(6, 11)
                shade = rng.randint(60, 130)
                img[r:r+h, c:c+w, :] = shade
        imgs.append(img)
    return imgs

def gen_farmland(n):
    """Striped green/yellow pattern mimicking crop rows."""
    imgs = []
    for _ in range(n):
        img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        stripe_w = rng.randint(4, 9)
        for col in range(SIZE):
            if (col // stripe_w) % 2 == 0:
                img[:, col, 1] = rng.randint(120, 190)   # green crop
                img[:, col, 0] = rng.randint(40,  90)
            else:
                img[:, col, 1] = rng.randint(160, 210)   # yellow soil
                img[:, col, 0] = rng.randint(140, 200)
                img[:, col, 2] = rng.randint(10,  40)
        imgs.append(img)
    return imgs

GENERATORS = [gen_water, gen_forest, gen_urban, gen_farmland]
N_PER_CLASS = 80   # 80 train + 20 test per class

def build_dataset(n_per_class, seed_offset=0):
    rng.seed(42 + seed_offset)
    images, labels = [], []
    for cls_idx, gen_fn in enumerate(GENERATORS):
        imgs = gen_fn(n_per_class)
        images.extend(imgs)
        labels.extend([cls_idx] * n_per_class)
    images = np.array(images, dtype=np.float32) / 255.0
    images = images.transpose(0, 3, 1, 2)   # NCHW for PyTorch
    labels = np.array(labels, dtype=np.int64)
    idx    = rng.permutation(len(labels))
    return images[idx], labels[idx]

# ─── CNN MODEL ────────────────────────────────────────────────────────────────

class LandCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32x32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ─── TRAINING ─────────────────────────────────────────────────────────────────

print("Satellite Land-Use Classifier")
print("=" * 50)
print("Generating synthetic patches (water/forest/urban/farmland)...")

X_train, y_train = build_dataset(N_PER_CLASS, seed_offset=0)
X_test,  y_test  = build_dataset(20, seed_offset=100)

train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=32)

model     = LandCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(f"Training: {len(train_ds)} samples | Test: {len(test_ds)} | 5 epochs")
t0 = time.time()
for epoch in range(5):
    model.train()
    total_loss, correct = 0.0, 0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct    += (pred.argmax(1) == yb).sum().item()
    acc = correct / len(train_ds)
    print(f"  Epoch {epoch+1}/5  loss={total_loss/len(train_ds):.4f}  train_acc={acc:.3f}")

# ─── EVALUATE ─────────────────────────────────────────────────────────────────
model.eval()
correct = 0
with torch.no_grad():
    for xb, yb in test_dl:
        correct += (model(xb).argmax(1) == yb).sum().item()
test_acc = correct / len(test_ds)
print(f"\nTest accuracy: {test_acc:.3f}  ({time.time()-t0:.1f}s)")

# ─── LAND-USE MAP VISUALISATION ───────────────────────────────────────────────
# Generate a 6x6 grid of patches, predict each, plot as a map

GRID   = 6
COLORS = {"water": "#2196F3", "forest": "#2E7D32",
          "urban": "#9E9E9E",  "farmland": "#FDD835"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: the raw patch grid
sample_patches = []
sample_labels  = []
model.eval()
pred_labels    = []
rng.seed(999)
for i in range(GRID * GRID):
    cls_idx  = rng.randint(0, 4)
    patch_np = GENERATORS[cls_idx](1)[0].astype(np.float32) / 255.0
    sample_patches.append(patch_np)
    sample_labels.append(cls_idx)
    patch_t = torch.tensor(patch_np.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        logits  = model(patch_t)
        pred_c  = logits.argmax(1).item()
    pred_labels.append(pred_c)

# Stitch patches into a mosaic
mosaic = np.zeros((GRID * SIZE, GRID * SIZE, 3), dtype=np.uint8)
for idx, patch in enumerate(sample_patches):
    r, c = divmod(idx, GRID)
    mosaic[r*SIZE:(r+1)*SIZE, c*SIZE:(c+1)*SIZE] = (patch * 255).astype(np.uint8)

axes[0].imshow(mosaic)
axes[0].set_title("Synthetic Satellite Mosaic (6x6 patches)")
axes[0].axis("off")

# Right: prediction map (each cell coloured by predicted class)
pred_arr = np.array(pred_labels).reshape(GRID, GRID)
color_mat = np.zeros((GRID * 10, GRID * 10, 3), dtype=np.float32)
for r in range(GRID):
    for c in range(GRID):
        lbl   = LABELS[pred_arr[r, c]]
        hex_c = COLORS[lbl].lstrip("#")
        rgb   = tuple(int(hex_c[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        color_mat[r*10:(r+1)*10, c*10:(c+1)*10, :] = rgb

axes[1].imshow(color_mat, interpolation="nearest")
# Overlay text labels
for r in range(GRID):
    for c in range(GRID):
        lbl   = LABELS[pred_arr[r, c]][0].upper()   # W/F/U/A first letter
        match = (pred_arr[r, c] == sample_labels[r * GRID + c])
        fc    = "white" if match else "red"
        axes[1].text(c * 10 + 5, r * 10 + 5, lbl,
                     ha="center", va="center", fontsize=7,
                     color=fc, fontweight="bold")
axes[1].set_title(f"CNN Prediction Map  (acc={test_acc:.2f})\nW=water F=forest U=urban A=farmland | red=wrong")
axes[1].axis("off")

legend_patches = [mpatches.Patch(color=v, label=k) for k, v in COLORS.items()]
axes[1].legend(handles=legend_patches, loc="lower right", fontsize=8,
               framealpha=0.8)

plt.suptitle("Satellite Land-Use CNN Classifier", fontsize=12)
plt.tight_layout()
plt.savefig("CNN/outputs/satellite_land_map.png", dpi=90)
plt.close()
print("Plot saved -> CNN/outputs/satellite_land_map.png")
print("[DONE] satellite_land_classifier.py complete")
