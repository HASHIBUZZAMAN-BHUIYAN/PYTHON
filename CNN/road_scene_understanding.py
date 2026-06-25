"""
road_scene_understanding.py
===========================
What it does:
    Generates synthetic 64x64 road-scene images using NumPy and Matplotlib,
    then trains a tiny PyTorch CNN to classify each image into one of 4 classes
    based on whether it contains a car and/or a road sign.

What it teaches:
    - Procedural image generation for dataset creation
    - Multi-class image classification with a small CNN
    - Per-class accuracy reporting
    - Forward pass and prediction visualization

RAM estimate  : ~300 MB peak
Time estimate : ~1-2 minutes on CPU
Real vs simulated: SIMULATED. All images are synthetically generated via NumPy
    drawing routines. No real road footage or camera input is used.
"""

import os
import numpy as np

# Must set Agg before any pyplot import (no display/GUI needed)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
IMG_SIZE   = 64
N_SAMPLES  = 300
N_EPOCHS   = 5
BATCH_SIZE = 32
SEED       = 42
OUTPUT_DIR = "CNN/outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -------------------------------------------------------------------------
# Synthetic road-scene generator
# -------------------------------------------------------------------------
def draw_road(img):
    """Fill the lower 60% of image with a dark-gray road."""
    road_top = int(IMG_SIZE * 0.40)
    img[road_top:, :] = 60   # dark gray road


def draw_lane_lines(img):
    """Draw two dashed yellow lane lines."""
    road_top = int(IMG_SIZE * 0.40)
    # Left lane line
    for y in range(road_top, IMG_SIZE, 6):
        img[y:y+3, int(IMG_SIZE * 0.30)] = 210
    # Right lane line
    for y in range(road_top, IMG_SIZE, 6):
        img[y:y+3, int(IMG_SIZE * 0.70)] = 210


def draw_sky(img):
    """Light-blue-ish sky in upper portion."""
    road_top = int(IMG_SIZE * 0.40)
    img[:road_top, :] = 180


def draw_car(img, rng):
    """Draw a rectangle 'car' on the road section."""
    road_top = int(IMG_SIZE * 0.40)
    car_w = rng.integers(10, 18)
    car_h = rng.integers(6, 12)
    car_x = rng.integers(5, IMG_SIZE - car_w - 5)
    car_y = rng.integers(road_top + 2, IMG_SIZE - car_h - 2)
    color = rng.integers(100, 220)
    img[car_y:car_y + car_h, car_x:car_x + car_w] = color
    # Windshield — darker strip
    img[car_y:car_y + 3, car_x + 2:car_x + car_w - 2] = 30


def draw_sign(img, rng):
    """Draw a small circular road sign on the roadside."""
    road_top = int(IMG_SIZE * 0.40)
    cx = rng.integers(5, 15)
    cy = rng.integers(road_top - 8, road_top + 5)
    radius = rng.integers(3, 6)
    # Rasterize a filled circle
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                py, px = cy + dy, cx + dx
                if 0 <= py < IMG_SIZE and 0 <= px < IMG_SIZE:
                    img[py, px] = 230  # bright sign


def generate_scene(has_car, has_sign, rng):
    """Generate a single 64x64 grayscale road scene as float32 [0,1]."""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    draw_sky(img)
    draw_road(img)
    draw_lane_lines(img)
    if has_car:
        draw_car(img, rng)
    if has_sign:
        draw_sign(img, rng)
    # Add slight noise
    noise = rng.integers(-8, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img.astype(np.float32) / 255.0


def build_dataset(n, rng):
    """Build dataset: label = 2*has_car + has_sign  -> 0,1,2,3."""
    images, labels = [], []
    per_class = n // 4
    for has_car in [0, 1]:
        for has_sign in [0, 1]:
            label = 2 * has_car + has_sign
            for _ in range(per_class):
                img = generate_scene(has_car, has_sign, rng)
                images.append(img)
                labels.append(label)
    images = np.array(images, dtype=np.float32)[:, np.newaxis, :, :]  # (N,1,H,W)
    labels = np.array(labels, dtype=np.int64)
    # Shuffle
    idx = rng.permutation(len(images))
    return images[idx], labels[idx]


# -------------------------------------------------------------------------
# Tiny CNN model  (~25k params)
# -------------------------------------------------------------------------
class RoadCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),   # 64x64 -> 64x64
            nn.ReLU(),
            nn.MaxPool2d(2),                  # -> 32x32
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # -> 16x16
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------
def train_model(model, loader, optimizer, criterion, n_epochs):
    model.train()
    for epoch in range(1, n_epochs + 1):
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
        avg_loss = total_loss / total
        acc = 100.0 * correct / total
        print(f"  Epoch {epoch}/{n_epochs}  loss={avg_loss:.4f}  acc={acc:.1f}%")


# -------------------------------------------------------------------------
# Per-class accuracy
# -------------------------------------------------------------------------
def per_class_accuracy(model, images_t, labels_t):
    model.eval()
    class_names = ["no-car no-sign", "no-car sign", "car no-sign", "car sign"]
    with torch.no_grad():
        preds = model(images_t).argmax(dim=1).numpy()
    labels = labels_t.numpy()
    print("\nPer-class accuracy:")
    for cls in range(4):
        mask = labels == cls
        if mask.sum() == 0:
            continue
        acc = 100.0 * (preds[mask] == cls).sum() / mask.sum()
        print(f"  [{cls}] {class_names[cls]:<20s}: {acc:.1f}%  ({mask.sum()} samples)")


# -------------------------------------------------------------------------
# Prediction grid plot
# -------------------------------------------------------------------------
def save_prediction_grid(model, images, labels, output_path):
    model.eval()
    images_t = torch.tensor(images[:16])
    with torch.no_grad():
        preds = model(images_t).argmax(dim=1).numpy()
    class_names = ["no-car/no-sign", "no-car/sign", "car/no-sign", "car/sign"]

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("Road Scene Predictions (first 16 samples)", fontsize=12)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, 0], cmap="gray", vmin=0, vmax=1)
        true_lbl = labels[i]
        pred_lbl = preds[i]
        color = "green" if pred_lbl == true_lbl else "red"
        ax.set_title(
            f"T:{class_names[true_lbl][:8]}\nP:{class_names[pred_lbl][:8]}",
            fontsize=6, color=color
        )
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=80)
    plt.close()
    print(f"\nPrediction grid saved -> {output_path}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Road Scene Understanding - Synthetic CNN Classifier")
    print("=" * 60)
    print(f"Generating {N_SAMPLES} synthetic road-scene images ...")

    rng = np.random.default_rng(SEED)
    images, labels = build_dataset(N_SAMPLES, rng)
    print(f"  Dataset shape: {images.shape}  labels: {labels.shape}")
    print(f"  Class counts: {np.bincount(labels)}")

    # Split 80/20
    split = int(0.8 * len(images))
    x_train = torch.tensor(images[:split])
    y_train = torch.tensor(labels[:split])
    x_test  = torch.tensor(images[split:])
    y_test  = torch.tensor(labels[split:])

    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model     = RoadCNN(n_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"\nModel parameters: {count_params(model):,}")
    print(f"\nTraining for {N_EPOCHS} epochs ...")
    train_model(model, train_loader, optimizer, criterion, N_EPOCHS)

    # Test accuracy
    model.eval()
    with torch.no_grad():
        test_preds = model(x_test).argmax(dim=1)
    test_acc = 100.0 * (test_preds == y_test).float().mean().item()
    print(f"\nTest accuracy: {test_acc:.1f}%")

    per_class_accuracy(model, x_test, y_test)

    output_path = os.path.join(OUTPUT_DIR, "road_scene_predictions.png")
    save_prediction_grid(model, images, labels, output_path)

    print("\n[OK] road_scene_understanding.py completed successfully.")


if __name__ == "__main__":
    main()
