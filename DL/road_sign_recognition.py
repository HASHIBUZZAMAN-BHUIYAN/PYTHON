"""
road_sign_recognition.py
========================
What it does:
    Procedurally generates synthetic 64x64 colour images of four traffic sign
    types (stop, yield, warning, speed-limit), then trains a small 3-layer CNN
    to classify them.

What it teaches:
    - Programmatic image generation with PIL
    - Data augmentation via random geometric variation at generation time
    - CNN design for image classification
    - Visualising per-class accuracy and a prediction grid

RAM estimate : ~80 MB
Time estimate: ~25 seconds on CPU (Ryzen 7)
Real vs simulated: ALL images are procedurally generated using PIL drawing
    primitives. No real traffic-sign photographs are downloaded or used.
"""

import os
import time
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

matplotlib_available = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib_available = False

os.makedirs("DL/outputs", exist_ok=True)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

IMG_SIZE    = 64
N_PER_CLASS = 100
CLASS_NAMES = ["stop", "yield", "warning", "speed_limit"]

# ── 1. Image generation helpers ───────────────────────────────────────────────

def rand_offset(rng_local, lo=-3, hi=3):
    return rng_local.randint(lo, hi)

def make_stop_sign(rng_local):
    """Red octagon with white border."""
    img  = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    cx, cy = IMG_SIZE // 2 + rand_offset(rng_local), IMG_SIZE // 2 + rand_offset(rng_local)
    r  = 24 + rng_local.randint(-2, 2)
    pts = []
    for k in range(8):
        angle = math.radians(k * 45 + 22.5)
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(pts, fill=(210, 30, 30), outline=(255, 255, 255))
    inner_r = r - 6
    inner_pts = [(cx + inner_r * math.cos(math.radians(k*45+22.5)),
                  cy + inner_r * math.sin(math.radians(k*45+22.5))) for k in range(8)]
    draw.polygon(inner_pts, fill=(210, 30, 30))
    return img

def make_yield_sign(rng_local):
    """Red downward-pointing triangle with white fill."""
    img  = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    cx   = IMG_SIZE // 2 + rand_offset(rng_local)
    top  = 10 + rand_offset(rng_local)
    bot  = IMG_SIZE - 8 + rand_offset(rng_local)
    half = 26 + rng_local.randint(-2, 2)
    outer = [(cx, bot), (cx - half, top), (cx + half, top)]
    draw.polygon(outer, fill=(210, 30, 30), outline=(210, 30, 30))
    margin = 5
    inner = [(cx, bot - margin), (cx - half + margin, top + margin),
             (cx + half - margin, top + margin)]
    draw.polygon(inner, fill=(255, 255, 255))
    return img

def make_warning_sign(rng_local):
    """Yellow circle with black border."""
    img  = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    cx, cy = IMG_SIZE // 2 + rand_offset(rng_local), IMG_SIZE // 2 + rand_offset(rng_local)
    r  = 22 + rng_local.randint(-2, 2)
    draw.ellipse([cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3],
                 fill=(0, 0, 0))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 fill=(255, 220, 0))
    draw.rectangle([cx - 2, cy - 8, cx + 2, cy + 2], fill=(0, 0, 0))
    draw.ellipse([cx - 2, cy + 5, cx + 2, cy + 9], fill=(0, 0, 0))
    return img

def make_speed_limit_sign(rng_local):
    """Blue rectangle with white number."""
    img  = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    cx, cy = IMG_SIZE // 2 + rand_offset(rng_local), IMG_SIZE // 2 + rand_offset(rng_local)
    w, h   = 36 + rng_local.randint(-2, 2), 44 + rng_local.randint(-2, 2)
    x0, y0, x1, y1 = cx - w//2, cy - h//2, cx + w//2, cy + h//2
    draw.rounded_rectangle([x0, y0, x1, y1], radius=4,
                            fill=(30, 80, 200), outline=(255, 255, 255))
    draw.rectangle([x0 + 3, y0 + 3, x1 - 3, y0 + 14], fill=(255, 255, 255))
    # Draw a digit "50" crudely using rectangles
    # digit '5'
    dx, dy = cx - 9, cy - 4
    draw.rectangle([dx, dy,      dx + 10, dy + 2],  fill=(255, 255, 255))
    draw.rectangle([dx, dy,      dx + 2,  dy + 8],  fill=(255, 255, 255))
    draw.rectangle([dx, dy + 6,  dx + 10, dy + 8],  fill=(255, 255, 255))
    draw.rectangle([dx + 8, dy + 8, dx + 10, dy + 16], fill=(255, 255, 255))
    draw.rectangle([dx, dy + 14, dx + 10, dy + 16], fill=(255, 255, 255))
    # digit '0'
    dx2 = cx + 2
    draw.rectangle([dx2,     dy,      dx2 + 10, dy + 2],  fill=(255, 255, 255))
    draw.rectangle([dx2,     dy + 14, dx2 + 10, dy + 16], fill=(255, 255, 255))
    draw.rectangle([dx2,     dy,      dx2 + 2,  dy + 16], fill=(255, 255, 255))
    draw.rectangle([dx2 + 8, dy,      dx2 + 10, dy + 16], fill=(255, 255, 255))
    return img

GENERATORS = [make_stop_sign, make_yield_sign, make_warning_sign, make_speed_limit_sign]

# ── 2. Build dataset ───────────────────────────────────────────────────────────
print("[OK] Generating synthetic road sign images...")
all_images, all_labels = [], []
for class_idx, gen_fn in enumerate(GENERATORS):
    rng_local = random.Random(class_idx * 100)
    for _ in range(N_PER_CLASS):
        img = gen_fn(rng_local)
        arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, C)
        all_images.append(arr)
        all_labels.append(class_idx)

X = np.stack(all_images).transpose(0, 3, 1, 2)  # (N, C, H, W)
y = np.array(all_labels, dtype=np.int64)

idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

X_t = torch.from_numpy(X)
y_t = torch.from_numpy(y)
print(f"[OK] Dataset: X={X_t.shape}  y unique={np.unique(y)}")

dataset = TensorDataset(X_t, y_t)
n_train = int(0.8 * len(dataset))
n_test  = len(dataset) - n_train
train_ds, test_ds = random_split(dataset, [n_train, n_test],
                                 generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

# ── 3. Small CNN ──────────────────────────────────────────────────────────────
class RoadSignCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)   # 64->64
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 32->32
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # 16->16
        self.pool  = nn.MaxPool2d(2, 2)
        self.bn1   = nn.BatchNorm2d(16)
        self.bn2   = nn.BatchNorm2d(32)
        self.bn3   = nn.BatchNorm2d(64)
        self.drop  = nn.Dropout(0.4)
        # After 3 pools: 64 -> 32 -> 16 -> 8  =>  64*8*8 = 4096
        self.fc1   = nn.Linear(64 * 8 * 8, 128)
        self.fc2   = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

model     = RoadSignCNN()
total_params = sum(p.numel() for p in model.parameters())
print(f"[OK] CNN params: {total_params:,}")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ── 4. Training ────────────────────────────────────────────────────────────────
EPOCHS = 10
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(xb)
        correct += (out.argmax(1) == yb).sum().item()
        total   += len(yb)
    if epoch == 1 or epoch % 2 == 0:
        print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={running_loss/total:.4f}  "
              f"acc={correct/total:.3f}")

elapsed = time.time() - t0
print(f"[OK] Training done in {elapsed:.1f}s")

# ── 5. Evaluation ─────────────────────────────────────────────────────────────
model.eval()
all_preds, all_true = [], []
test_images_sample, test_labels_sample, test_preds_sample = [], [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb).argmax(1)
        all_preds.extend(preds.numpy())
        all_true.extend(yb.numpy())
        if len(test_images_sample) < 16:
            test_images_sample.extend(xb.numpy())
            test_labels_sample.extend(yb.numpy())
            test_preds_sample.extend(preds.numpy())

all_preds = np.array(all_preds)
all_true  = np.array(all_true)
overall   = (all_preds == all_true).mean()
print(f"\n[OK] Overall test accuracy: {overall:.3f}")

for i, name in enumerate(CLASS_NAMES):
    mask = all_true == i
    if mask.sum() > 0:
        acc_i = (all_preds[mask] == all_true[mask]).mean()
        print(f"  {name:12s}: {acc_i:.3f}  (n={mask.sum()})")

# ── 6. Sample predictions grid ────────────────────────────────────────────────
if matplotlib_available:
    n_show = min(16, len(test_images_sample))
    cols   = 4
    rows   = math.ceil(n_show / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()
    for i in range(n_show):
        img_disp = np.transpose(test_images_sample[i], (1, 2, 0))
        axes[i].imshow(img_disp)
        true_lbl = CLASS_NAMES[test_labels_sample[i]]
        pred_lbl = CLASS_NAMES[test_preds_sample[i]]
        correct_flag = "[OK]" if true_lbl == pred_lbl else "[X]"
        axes[i].set_title(f"{correct_flag}\nT:{true_lbl}\nP:{pred_lbl}", fontsize=7)
        axes[i].axis("off")
    for i in range(n_show, len(axes)):
        axes[i].axis("off")
    plt.suptitle(f"Road Sign Predictions (Acc={overall:.3f})", y=1.01)
    plt.tight_layout()
    plt.savefig("DL/outputs/road_signs.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("[OK] Sample grid saved to DL/outputs/road_signs.png")

print("\n[DONE] road_sign_recognition.py complete")
