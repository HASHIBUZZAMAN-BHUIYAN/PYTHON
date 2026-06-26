"""
Wildlife Species Classifier
=============================
What it does:
  Generates synthetic animal silhouette images for 3 species categories:
    - bird     : small oval body + V-shaped wings + thin beak
    - deer     : taller oval body + 4 stick legs + branched antlers
    - rabbit   : round body + long upright ears + small tail
  Trains a small CNN classifier and visualises predictions.

  IMPORTANT DISCLAIMER: This is a teaching demo only. The "species" are
  simple geometric silhouettes drawn with OpenCV, not real animal photos.
  Real camera-trap wildlife-monitoring CNNs (e.g. Wildlife Insights,
  iNaturalist models) are trained on millions of actual wildlife photos
  and use ResNet-50 or EfficientNet with transfer learning. This demo
  shows the SAME classification pipeline at a toy scale so the concept
  is understandable without any downloads.

What it teaches:
  - How shape-based features allow a CNN to distinguish species silhouettes
  - Camera-trap deployment model: classify each triggered image into
    "species X present / absent" categories
  - The gap between this demo and real systems: real images have background
    clutter, varying lighting, partial occlusions -- which is why large
    datasets and deeper networks are necessary in practice
  - Confidence thresholding: outputs below e.g. 0.60 confidence would be
    sent to a human reviewer in a real system

How to run:
  python CNN\wildlife_species_classifier.py   (from PYTHON\ folder)

Estimated RAM: ~200MB | Time: ~20s
Model: 3-conv CNN, ~100k params. Fully synthetic silhouettes. No download.
Output: CNN\outputs\wildlife_species.png
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

SIZE   = 64
LABELS = ["bird", "deer", "rabbit"]

# ─── SILHOUETTE GENERATORS ────────────────────────────────────────────────────

def make_bird(bg=220):
    img    = np.full((SIZE, SIZE, 3), bg, dtype=np.uint8)
    colour = (30, 30, 30)        # dark silhouette
    cx, cy = SIZE//2 + rng.randint(-4, 5), SIZE//2 + rng.randint(-4, 5)
    # Body oval
    cv2.ellipse(img, (cx, cy), (10, 7), 0, 0, 360, colour, -1)
    # Head
    cv2.circle(img, (cx + 10, cy - 3), 5, colour, -1)
    # Beak (short horizontal line)
    cv2.line(img, (cx + 15, cy - 3), (cx + 20, cy - 3), colour, 2)
    # Wings: two triangular arcs
    wing_w = rng.randint(12, 18)
    pts_l  = np.array([[cx, cy - 2], [cx - wing_w, cy - 8], [cx - 5, cy + 4]], np.int32)
    pts_r  = np.array([[cx, cy - 2], [cx + wing_w//2, cy - 8], [cx + 3, cy + 4]], np.int32)
    cv2.fillPoly(img, [pts_l], colour)
    cv2.fillPoly(img, [pts_r], colour)
    return img


def make_deer(bg=220):
    img    = np.full((SIZE, SIZE, 3), bg, dtype=np.uint8)
    colour = (80, 50, 20)        # brownish silhouette
    cx, cy = SIZE//2 + rng.randint(-4, 5), SIZE//2 + rng.randint(-4, 5)
    # Body oval (wider than tall)
    cv2.ellipse(img, (cx, cy), (16, 10), 0, 0, 360, colour, -1)
    # Neck
    cv2.rectangle(img, (cx + 8, cy - 14), (cx + 12, cy), colour, -1)
    # Head
    cv2.ellipse(img, (cx + 12, cy - 17), (6, 5), 0, 0, 360, colour, -1)
    # 4 legs (thin rectangles downward)
    for lx in [cx - 10, cx - 4, cx + 4, cx + 10]:
        cv2.rectangle(img, (lx, cy + 10), (lx + 2, cy + 22), colour, -1)
    # Antlers (branching lines from head)
    hx, hy = cx + 12, cy - 22
    cv2.line(img, (hx, hy), (hx - 4, hy - 8), colour, 2)
    cv2.line(img, (hx - 4, hy - 8), (hx - 7, hy - 14), colour, 2)
    cv2.line(img, (hx - 4, hy - 8), (hx - 1, hy - 13), colour, 2)
    cv2.line(img, (hx, hy), (hx + 4, hy - 8), colour, 2)
    cv2.line(img, (hx + 4, hy - 8), (hx + 7, hy - 14), colour, 2)
    return img


def make_rabbit(bg=220):
    img    = np.full((SIZE, SIZE, 3), bg, dtype=np.uint8)
    colour = (150, 140, 130)     # light grey silhouette
    cx, cy = SIZE//2 + rng.randint(-4, 5), SIZE//2 + rng.randint(-4, 5)
    # Round body
    cv2.ellipse(img, (cx, cy), (13, 12), 0, 0, 360, colour, -1)
    # Head (slightly smaller circle)
    hx, hy = cx + 10, cy - 10
    cv2.circle(img, (hx, hy), 8, colour, -1)
    # Long upright ears
    cv2.ellipse(img, (hx - 3, hy - 14), (3, 10), 0, 0, 360, colour, -1)
    cv2.ellipse(img, (hx + 3, hy - 14), (3, 10), 0, 0, 360, colour, -1)
    # Inner ear (lighter)
    inner = tuple(min(255, c + 40) for c in colour)
    cv2.ellipse(img, (hx - 3, hy - 14), (1, 7), 0, 0, 360, inner, -1)
    cv2.ellipse(img, (hx + 3, hy - 14), (1, 7), 0, 0, 360, inner, -1)
    # Small tail
    cv2.circle(img, (cx - 14, cy), 4, (200, 200, 200), -1)
    # 2 front legs
    cv2.rectangle(img, (cx + 4, cy + 10), (cx + 6, cy + 18), colour, -1)
    cv2.rectangle(img, (cx - 2, cy + 12), (cx, cy + 18), colour, -1)
    return img


GENERATORS = [make_bird, make_deer, make_rabbit]
N_PER_CLASS = 100

def build_dataset(n):
    images, labels = [], []
    for cls_idx, gen_fn in enumerate(GENERATORS):
        for _ in range(n):
            img = gen_fn()
            img = np.clip(img.astype(int) + rng.randint(-8, 9, img.shape), 0, 255).astype(np.uint8)
            images.append(img)
            labels.append(cls_idx)
    images = np.array(images, dtype=np.float32) / 255.0
    images = images.transpose(0, 3, 1, 2)
    labels = np.array(labels, dtype=np.int64)
    idx    = rng.permutation(len(labels))
    return images[idx], labels[idx]

# ─── MODEL ────────────────────────────────────────────────────────────────────

class WildlifeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 3),
        )
    def forward(self, x): return self.net(x)

# ─── TRAIN ────────────────────────────────────────────────────────────────────

print("Wildlife Species Classifier")
print("=" * 50)
print("NOTE: synthetic silhouettes only (not real animal photos)")
print("Generating bird / deer / rabbit silhouettes...")

X_train, y_train = build_dataset(N_PER_CLASS)
X_test,  y_test  = build_dataset(25)

train_dl = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                      batch_size=32, shuffle=True)
test_dl  = DataLoader(TensorDataset(torch.tensor(X_test),  torch.tensor(y_test)),
                      batch_size=32)

model     = WildlifeCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

t0 = time.time()
EPOCHS = 8
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

# ─── VISUALISE ────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 8, figsize=(15, 6))
fig.suptitle(f"Wildlife Species Classifier  (test acc={test_acc:.2f})  "
             "[synthetic silhouettes, not real photos]\n"
             "green=correct red=wrong | conf shown as decimal", fontsize=9)

model.eval()
rng.seed(42)
for row, (gen_fn, cls_true) in enumerate(zip(GENERATORS, range(3))):
    axes[row][0].set_ylabel(f"True:\n{LABELS[cls_true]}", fontsize=8)
    for col in range(8):
        ax  = axes[row][col]
        img = gen_fn()
        img_t = torch.tensor(img.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
        with torch.no_grad():
            logits = model(img_t)
            probs  = torch.softmax(logits, 1).numpy()[0]
            pred   = logits.argmax(1).item()
        fc = "green" if pred == cls_true else "red"
        ax.imshow(img)
        ax.set_title(f"{LABELS[pred]}\n{probs[pred]:.2f}", fontsize=6.5, color=fc)
        ax.axis("off")

plt.tight_layout()
plt.savefig("CNN/outputs/wildlife_species.png", dpi=90)
plt.close()
print("Plot saved -> CNN/outputs/wildlife_species.png")
print()
print("NOTE: Real wildlife CNNs:")
print("  - Use millions of actual camera-trap photos (e.g. iWildCam dataset)")
print("  - Apply ResNet-50 or EfficientNet with ImageNet transfer learning")
print("  - Handle low-light, partial occlusion, motion blur")
print("  - This demo mirrors the SAME classification pipeline at toy scale")
print("[DONE] wildlife_species_classifier.py complete")
