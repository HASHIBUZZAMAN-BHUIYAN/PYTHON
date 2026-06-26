"""
Retail Product Shape Classifier
=================================
What it does:
  Generates synthetic product silhouette images for 3 retail shape categories:
    - bottle : tall narrow rectangle + circle cap
    - box    : square/rectangle with ruled edges
    - can    : shorter cylinder-style oval shape
  Trains a small CNN to classify product type from the silhouette.
  Outputs a test prediction grid and a "shelf view" of random products.

  Real-world mirror: retail shelf inventory systems (e.g. Amazon Go,
  automated warehouse robots) use CNNs to identify product types, brands,
  and facings from camera feeds. Real systems use large labelled datasets
  of actual product photos. This demo builds the same pipeline using
  simple shape silhouettes to keep it fully offline and explainable.

What it teaches:
  - How shape/silhouette is often the primary discriminating feature in
    object recognition before texture/colour
  - Why aspect ratio and contour geometry are strong classification signals
  - Multi-class CNN with 3 outputs and softmax probabilities
  - How a "shelf view" (multiple objects per image row) differs from
    single-object patch classification

How to run:
  python CNN\retail_product_classifier.py   (from PYTHON\ folder)

Estimated RAM: ~200MB | Time: ~20s
Model: 3-conv CNN, ~100k params. Fully synthetic silhouettes. No download.
Output: CNN\outputs\retail_shelf.png
"""

import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

os.makedirs("CNN/outputs", exist_ok=True)
rng = np.random.RandomState(42)
torch.manual_seed(42)

SIZE = 64
LABELS = ["bottle", "box", "can"]

# ─── SYNTHETIC SILHOUETTE GENERATORS ─────────────────────────────────────────

def make_bottle(bg=240):
    """Tall narrow rectangle with a smaller cap rectangle on top."""
    img = np.full((SIZE, SIZE, 3), bg, dtype=np.uint8)
    # Choose random colour for the bottle
    colour = tuple(int(c) for c in rng.randint(40, 200, 3))
    # Body: tall narrow rectangle
    bx1 = rng.randint(20, 26); bx2 = rng.randint(38, 44)
    by1 = rng.randint(20, 28); by2 = rng.randint(52, 60)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), colour, -1)
    # Cap: narrower, on top
    cx1 = (bx1 + bx2) // 2 - 4; cx2 = cx1 + 8
    cv2.rectangle(img, (cx1, by1 - 8), (cx2, by1), colour, -1)
    # Highlight line
    cv2.line(img, (bx1+2, by1+4), (bx1+2, by2-4), (min(255, colour[0]+60),)*3, 1)
    return img

def make_box(bg=240):
    """Square/rectangular box with corner lines."""
    img = np.full((SIZE, SIZE, 3), bg, dtype=np.uint8)
    colour = tuple(int(c) for c in rng.randint(40, 200, 3))
    x1 = rng.randint(12, 20); x2 = rng.randint(44, 52)
    y1 = rng.randint(12, 20); y2 = rng.randint(44, 52)
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, -1)
    # Box edge lines to give 3D feel
    dark = tuple(max(0, c - 40) for c in colour)
    cv2.line(img, (x1, y1), (x2, y1), dark, 2)
    cv2.line(img, (x1, y2), (x2, y2), dark, 2)
    cv2.line(img, (x1, y1), (x1, y2), dark, 2)
    cv2.line(img, (x2, y1), (x2, y2), dark, 2)
    return img

def make_can(bg=240):
    """Shorter oval/ellipse shape mimicking a can."""
    img = np.full((SIZE, SIZE, 3), bg, dtype=np.uint8)
    colour = tuple(int(c) for c in rng.randint(40, 200, 3))
    cx = SIZE // 2 + rng.randint(-4, 5)
    cy = SIZE // 2 + rng.randint(-4, 5)
    rx = rng.randint(14, 18)
    ry = rng.randint(20, 26)
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, colour, -1)
    # Top rim (flat line)
    top_y = cy - ry
    cv2.line(img, (cx - rx, top_y), (cx + rx, top_y),
             tuple(min(255, c + 50) for c in colour), 2)
    # Vertical highlight
    cv2.line(img, (cx - rx//3, cy - ry + 4), (cx - rx//3, cy + ry - 4),
             (min(255, colour[0] + 70),)*3, 1)
    return img

GENERATORS = [make_bottle, make_box, make_can]
N_PER_CLASS = 100

def build_dataset(n):
    images, labels = [], []
    for cls_idx, gen_fn in enumerate(GENERATORS):
        for _ in range(n):
            img = gen_fn()
            # Add slight noise and random background variation
            img = np.clip(img.astype(int) + rng.randint(-10, 11, img.shape), 0, 255).astype(np.uint8)
            images.append(img)
            labels.append(cls_idx)
    images = np.array(images, dtype=np.float32) / 255.0
    images = images.transpose(0, 3, 1, 2)
    labels = np.array(labels, dtype=np.int64)
    idx    = rng.permutation(len(labels))
    return images[idx], labels[idx]

# ─── MODEL ────────────────────────────────────────────────────────────────────

class ProductCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )
    def forward(self, x): return self.net(x)

# ─── TRAIN ────────────────────────────────────────────────────────────────────

print("Retail Product Classifier")
print("=" * 50)
print("Generating synthetic product silhouettes (bottle / box / can)...")

X_train, y_train = build_dataset(N_PER_CLASS)
X_test,  y_test  = build_dataset(25)

train_dl = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                      batch_size=32, shuffle=True)
test_dl  = DataLoader(TensorDataset(torch.tensor(X_test),  torch.tensor(y_test)),
                      batch_size=32)

model     = ProductCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

t0 = time.time()
EPOCHS = 6
for epoch in range(EPOCHS):
    model.train()
    tot_loss, correct = 0.0, 0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * len(yb)
        correct  += (out.argmax(1) == yb).sum().item()
    acc = correct / len(X_train)
    print(f"  Epoch {epoch+1}/{EPOCHS}  loss={tot_loss/len(X_train):.4f}  acc={acc:.3f}")

model.eval()
correct = 0
with torch.no_grad():
    for xb, yb in test_dl:
        correct += (model(xb).argmax(1) == yb).sum().item()
test_acc = correct / len(X_test)
print(f"\nTest accuracy: {test_acc:.3f}  ({time.time()-t0:.1f}s)")

# ─── VISUALISATION: prediction grid + "shelf view" ───────────────────────────

fig, axes = plt.subplots(3, 9, figsize=(16, 6))
fig.suptitle(f"Retail Product Classifier (test acc={test_acc:.2f})\n"
             "Each row = one product type  |  green=correct red=wrong", fontsize=10)

model.eval()
rng.seed(77)
col_labels = {0: "Bottle", 1: "Box", 2: "Can"}
for row, (gen_fn, cls_true) in enumerate(zip(GENERATORS, range(3))):
    for col in range(9):
        ax  = axes[row][col]
        img = gen_fn()
        img_t = torch.tensor(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            logits = model(img_t)
            probs  = torch.softmax(logits, 1).numpy()[0]
            pred   = logits.argmax(1).item()
        fc = "green" if pred == cls_true else "red"
        ax.imshow(img)
        ax.set_title(f"{LABELS[pred]}\n{probs[pred]:.2f}", fontsize=6, color=fc)
        ax.axis("off")
    axes[row][0].set_ylabel(f"True:\n{LABELS[cls_true]}", fontsize=8)

plt.tight_layout()
plt.savefig("CNN/outputs/retail_shelf.png", dpi=90)
plt.close()
print("Plot saved -> CNN/outputs/retail_shelf.png")
print("[DONE] retail_product_classifier.py complete")
