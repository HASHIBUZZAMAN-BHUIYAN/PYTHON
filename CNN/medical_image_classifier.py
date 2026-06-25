"""
medical_image_classifier.py
============================
What it does:
    Generates 400 synthetic 64x64 grayscale medical-style images:
      - "normal"   : uniform gray background + Gaussian noise only
      - "abnormal" : same background + a bright Gaussian blob (simulated lesion)
                     + noise
    Trains a tiny 3-conv-layer CNN for binary classification, then reports
    accuracy, sensitivity, specificity, and a confusion matrix plot alongside
    sample images.

What it teaches:
    - Synthetic medical image generation
    - Binary classification with CNN
    - Medical evaluation metrics: sensitivity (recall) and specificity
    - Confusion matrix interpretation

RAM estimate  : ~250 MB peak
Time estimate : ~1-2 minutes on CPU
Real vs simulated: SIMULATED. All images are procedurally generated.
    No real medical imaging data is used at any point.

DISCLAIMER: EDUCATIONAL EXAMPLE ONLY - NOT A REAL DIAGNOSTIC TOOL.
    This script is for learning CNN concepts. It must never be used for
    any form of real medical diagnosis or clinical decision-making.
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

OUTPUT_DIR = "CNN/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE   = 64
N_SAMPLES  = 400    # 200 normal + 200 abnormal
N_EPOCHS   = 8
BATCH_SIZE = 32
SEED       = 2024

np.random.seed(SEED)
torch.manual_seed(SEED)


# -------------------------------------------------------------------------
# Synthetic image generators
# -------------------------------------------------------------------------
def gaussian_blob(size, cx, cy, sigma):
    """Return a 2-D Gaussian blob as float32 array in [0, 1]."""
    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)
    blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return blob.astype(np.float32)


def make_normal_image(rng):
    """Uniform gray + small Gaussian noise -> 'healthy tissue'."""
    base = np.full((IMG_SIZE, IMG_SIZE), 0.45, dtype=np.float32)
    noise = rng.standard_normal((IMG_SIZE, IMG_SIZE)).astype(np.float32) * 0.05
    return np.clip(base + noise, 0.0, 1.0)


def make_abnormal_image(rng):
    """Uniform gray + bright blob (lesion) + Gaussian noise -> 'lesion'."""
    base = np.full((IMG_SIZE, IMG_SIZE), 0.45, dtype=np.float32)
    # Random lesion position and size
    cx = rng.integers(16, IMG_SIZE - 16)
    cy = rng.integers(16, IMG_SIZE - 16)
    sigma = rng.uniform(4.0, 9.0)
    intensity = rng.uniform(0.3, 0.5)
    blob = gaussian_blob(IMG_SIZE, cx, cy, sigma)
    noise = rng.standard_normal((IMG_SIZE, IMG_SIZE)).astype(np.float32) * 0.05
    return np.clip(base + intensity * blob + noise, 0.0, 1.0)


def build_dataset(rng):
    images, labels = [], []
    half = N_SAMPLES // 2
    for _ in range(half):
        images.append(make_normal_image(rng))
        labels.append(0)
    for _ in range(half):
        images.append(make_abnormal_image(rng))
        labels.append(1)
    images = np.array(images, dtype=np.float32)[:, np.newaxis, :, :]
    labels = np.array(labels, dtype=np.int64)
    idx = rng.permutation(len(images))
    return images[idx], labels[idx]


# -------------------------------------------------------------------------
# Tiny CNN  (~35k params)
# -------------------------------------------------------------------------
class MedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 32x32
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 16x16
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
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
# Metrics
# -------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    """Return accuracy, sensitivity (recall for class 1), specificity."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    acc         = (tp + tn) / len(y_true) * 100
    sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
    confusion   = np.array([[tn, fp], [fn, tp]], dtype=int)
    return acc, sensitivity, specificity, confusion


# -------------------------------------------------------------------------
# Combined output plot
# -------------------------------------------------------------------------
def save_output_plot(model, images, labels, confusion, output_path):
    model.eval()
    # ---- Row 1: sample images ----
    # Pick 4 normal and 4 abnormal for display
    normal_idx   = np.where(labels == 0)[0][:4]
    abnormal_idx = np.where(labels == 1)[0][:4]
    sample_idx   = np.concatenate([normal_idx, abnormal_idx])
    sample_imgs  = images[sample_idx]
    sample_lbls  = labels[sample_idx]
    sample_names = ["Normal"] * 4 + ["Abnormal"] * 4

    x_s = torch.tensor(sample_imgs)
    with torch.no_grad():
        preds_s = model(x_s).argmax(dim=1).numpy()
    pred_names = ["Normal" if p == 0 else "Abnormal" for p in preds_s]

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle("Medical Image Classifier  --  EDUCATIONAL EXAMPLE ONLY", fontsize=10)

    gs = fig.add_gridspec(2, 9, hspace=0.4, wspace=0.3)

    # Top row: 8 sample images
    for i in range(8):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(sample_imgs[i, 0], cmap="gray", vmin=0, vmax=1)
        color = "green" if pred_names[i] == sample_names[i] else "red"
        ax.set_title(f"T:{sample_names[i][:3]}\nP:{pred_names[i][:3]}",
                     fontsize=6, color=color)
        ax.axis("off")

    # Bottom row: confusion matrix (spans full width)
    ax_cm = fig.add_subplot(gs[1, 3:6])
    im = ax_cm.imshow(confusion, cmap="Blues", vmin=0)
    class_labels = ["Normal", "Abnormal"]
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(class_labels, fontsize=9)
    ax_cm.set_yticklabels(class_labels, fontsize=9)
    ax_cm.set_xlabel("Predicted", fontsize=9)
    ax_cm.set_ylabel("Actual", fontsize=9)
    ax_cm.set_title("Confusion Matrix", fontsize=10)
    for r in range(2):
        for c in range(2):
            ax_cm.text(c, r, str(confusion[r, c]),
                       ha="center", va="center", fontsize=14, fontweight="bold")

    plt.savefig(output_path, dpi=80, bbox_inches="tight")
    plt.close()
    print(f"\nOutput plot saved -> {output_path}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Medical Image Classifier - Synthetic Lesion Detection")
    print("=" * 60)
    print("DISCLAIMER: EDUCATIONAL EXAMPLE ONLY - NOT A REAL DIAGNOSTIC TOOL")
    print()

    rng = np.random.default_rng(SEED)
    print(f"Generating {N_SAMPLES} synthetic medical images ...")
    images, labels = build_dataset(rng)
    print(f"  Dataset: {images.shape}  normal={np.sum(labels==0)}  abnormal={np.sum(labels==1)}")

    split = int(0.8 * len(images))
    x_train = torch.tensor(images[:split])
    y_train = torch.tensor(labels[:split])
    x_test  = torch.tensor(images[split:])
    y_test  = torch.tensor(labels[split:])

    train_ds     = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model     = MedCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"\nModel parameters: {count_params(model):,}")
    print(f"\nTraining for {N_EPOCHS} epochs ...")
    train_model(model, train_loader, optimizer, criterion, N_EPOCHS)

    model.eval()
    with torch.no_grad():
        test_preds = model(x_test).argmax(dim=1).numpy()
    y_true = y_test.numpy()

    acc, sensitivity, specificity, confusion = compute_metrics(y_true, test_preds)
    print(f"\nTest Results:")
    print(f"  Accuracy    : {acc:.1f}%")
    print(f"  Sensitivity : {sensitivity:.1f}%  (abnormal detected correctly)")
    print(f"  Specificity : {specificity:.1f}%  (normal detected correctly)")
    print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
    print(f"              Normal  Abnormal")
    print(f"  Normal    :   {confusion[0,0]:4d}    {confusion[0,1]:4d}")
    print(f"  Abnormal  :   {confusion[1,0]:4d}    {confusion[1,1]:4d}")

    output_path = os.path.join(OUTPUT_DIR, "medical_classifier.png")
    save_output_plot(model, images, labels, confusion, output_path)

    print("\n[OK] medical_image_classifier.py completed successfully.")
    print("REMINDER: EDUCATIONAL EXAMPLE ONLY - NOT A REAL DIAGNOSTIC TOOL")


if __name__ == "__main__":
    main()
