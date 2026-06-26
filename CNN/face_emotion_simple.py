"""
face_emotion_simple.py
======================
What it does:
    1. Generates ~200 synthetic grayscale face images (64x64) using PIL:
       circle head, two eye dots, and either an upward curve (happy) or a
       downward curve (sad) for the mouth.
    2. Demonstrates Haar cascade face detection on a sample synthetic face
       using OpenCV's built-in frontal-face cascade (no download required -
       the XML file ships with the opencv-python package).
    3. Trains a tiny 2-conv-layer CNN to classify happy vs sad faces.
    4. Reports accuracy and saves sample predictions to an output grid.

What it teaches:
    - Synthetic image generation with PIL drawing primitives
    - Classic Haar cascade face detection (cv2.CascadeClassifier)
    - Binary image classification with a small CNN
    - Happy/sad emotion classification pipeline

RAM estimate  : ~200 MB peak
Time estimate : ~30-60 seconds on CPU
Real vs simulated: SIMULATED. Synthetic face drawings, not real faces.
    Educational only. The Haar cascade ships with opencv-python (no download).

NOTE: "Synthetic face drawings, not real faces. Educational only."
"""

import os
import math
import numpy as np
import cv2

from PIL import Image, ImageDraw

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
N_SAMPLES  = 200   # 100 happy + 100 sad
N_EPOCHS   = 5
BATCH_SIZE = 32
SEED       = 99

np.random.seed(SEED)
torch.manual_seed(SEED)


# -------------------------------------------------------------------------
# Synthetic face generator (PIL)
# -------------------------------------------------------------------------
def draw_face(label, rng):
    """
    Draw a simple grayscale cartoon face.
    label=0 -> sad  (downward mouth curve)
    label=1 -> happy (upward mouth curve)
    Returns a (64, 64) uint8 numpy array.
    """
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=30)   # dark background
    draw = ImageDraw.Draw(img)

    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    r_head = 24
    draw.ellipse(
        [cx - r_head, cy - r_head, cx + r_head, cy + r_head],
        fill=180, outline=220
    )

    eye_y  = cy - 8
    eye_dx = 9
    r_eye  = 3
    draw.ellipse([cx - eye_dx - r_eye, eye_y - r_eye,
                  cx - eye_dx + r_eye, eye_y + r_eye], fill=30)
    draw.ellipse([cx + eye_dx - r_eye, eye_y - r_eye,
                  cx + eye_dx + r_eye, eye_y + r_eye], fill=30)

    mouth_l = cx - 10
    mouth_r = cx + 10
    mouth_top = cy + 5
    mouth_bot = cy + 14

    if label == 1:  # happy - arc opens upward, drawn as bottom arc
        draw.arc([mouth_l, mouth_top, mouth_r, mouth_bot],
                 start=0, end=180, fill=50, width=2)
    else:           # sad - arc opens downward, drawn as top arc
        draw.arc([mouth_l, mouth_top, mouth_r, mouth_bot],
                 start=180, end=360, fill=50, width=2)

    arr = np.array(img, dtype=np.float32) / 255.0

    noise_level = rng.uniform(0.0, 0.04)
    arr = np.clip(arr + rng.standard_normal(arr.shape).astype(np.float32) * noise_level,
                  0.0, 1.0)
    return arr


def build_face_dataset(rng):
    images, labels = [], []
    for label in [0, 1]:       # 0=sad, 1=happy
        for _ in range(N_SAMPLES // 2):
            img = draw_face(label, rng)
            images.append(img[np.newaxis, :, :])  # (1, H, W)
            labels.append(label)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    idx = rng.permutation(len(images))
    return images[idx], labels[idx]


# -------------------------------------------------------------------------
# Haar cascade demo
# -------------------------------------------------------------------------
def demo_haar_cascade():
    """
    Create a simple synthetic face image and run Haar cascade detection on it.
    The cascade XML is bundled with opencv-python, no download needed.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        print("  [!] Haar cascade XML not found at:", cascade_path)
        return

    cascade = cv2.CascadeClassifier(cascade_path)

    img_pil = Image.new("L", (96, 96), color=30)
    draw = ImageDraw.Draw(img_pil)
    draw.ellipse([16, 12, 80, 76], fill=180, outline=220)
    draw.ellipse([31, 29, 37, 35], fill=40)
    draw.ellipse([59, 29, 65, 35], fill=40)
    draw.arc([35, 48, 61, 64], start=0, end=180, fill=60, width=2)
    img_np = np.array(img_pil, dtype=np.uint8)

    faces = cascade.detectMultiScale(img_np, scaleFactor=1.1, minNeighbors=3,
                                     minSize=(20, 20))
    n_faces = len(faces) if faces is not None and len(faces) > 0 else 0
    print(f"  Haar cascade demo: {n_faces} face region(s) detected "
          f"in synthetic 96x96 image")
    if n_faces == 0:
        print("  (0 detections is normal - Haar cascades are tuned for real photos)")


# -------------------------------------------------------------------------
# Tiny emotion CNN  (~8k params)
# -------------------------------------------------------------------------
class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),   # -> 64x64
            nn.ReLU(),
            nn.MaxPool2d(2),                  # -> 32x32
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # -> 16x16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
# Prediction grid
# -------------------------------------------------------------------------
def save_prediction_grid(model, images, labels, output_path):
    model.eval()
    n = 16
    x = torch.tensor(images[:n])
    with torch.no_grad():
        preds = model(x).argmax(dim=1).numpy()
    class_names = ["Sad", "Happy"]

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("Face Emotion Predictions (Synthetic Faces)", fontsize=11)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, 0], cmap="gray", vmin=0, vmax=1)
        true_lbl = labels[i]
        pred_lbl = preds[i]
        color = "green" if pred_lbl == true_lbl else "red"
        ax.set_title(f"T:{class_names[true_lbl]} P:{class_names[pred_lbl]}",
                     fontsize=7, color=color)
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
    print("Face Emotion Classifier - Synthetic Happy/Sad CNN")
    print("=" * 60)
    print("NOTE: Synthetic face drawings, not real faces. Educational only.")
    print()

    print("Haar Cascade Demo:")
    demo_haar_cascade()

    rng = np.random.default_rng(SEED)
    print(f"\nGenerating {N_SAMPLES} synthetic face images ...")
    images, labels = build_face_dataset(rng)
    print(f"  Dataset shape: {images.shape}  labels: {labels.shape}")
    print(f"  Class counts: sad={np.sum(labels==0)}, happy={np.sum(labels==1)}")

    split = int(0.8 * len(images))
    x_train = torch.tensor(images[:split])
    y_train = torch.tensor(labels[:split])
    x_test  = torch.tensor(images[split:])
    y_test  = torch.tensor(labels[split:])

    train_ds     = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model     = EmotionCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"\nModel parameters: {count_params(model):,}")
    print(f"\nTraining for {N_EPOCHS} epochs ...")
    train_model(model, train_loader, optimizer, criterion, N_EPOCHS)

    model.eval()
    with torch.no_grad():
        test_preds = model(x_test).argmax(dim=1)
    test_acc = 100.0 * (test_preds == y_test).float().mean().item()
    print(f"\nTest accuracy: {test_acc:.1f}%")

    output_path = os.path.join(OUTPUT_DIR, "face_emotion.png")
    save_prediction_grid(model, images, labels, output_path)

    print("\n[OK] face_emotion_simple.py completed successfully.")


if __name__ == "__main__":
    main()
