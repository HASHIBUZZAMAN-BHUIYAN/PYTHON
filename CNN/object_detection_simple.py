"""
Simple Object Detector (Sliding-Window CNN)
=============================================
What it does:
  Implements a simplified object detector on synthetic images. Each image
  contains 1-3 simple shapes (filled circles or filled squares) placed at
  random positions on a white background.

  Approach (sliding window):
    1. Train a small "patch classifier" CNN on 32x32 crops: shape-present
       vs background (no shape).
    2. At detection time, slide the window across the image at multiple
       scales, collecting high-confidence "shape present" detections.
    3. Apply Non-Maximum Suppression (NMS) to remove overlapping boxes.
    4. Draw predicted bounding boxes on the output image.

  This is the same conceptual pipeline as early R-CNN and DPM detectors.
  Modern YOLO/SSD replace the sliding window with a single-pass grid CNN,
  but the underlying idea is identical: classify each region and NMS.

What it teaches:
  - Object detection vs classification: "what is it?" + "where is it?"
  - Sliding window: exhaustive but slow; grid anchor approach fixes the speed
  - Non-Maximum Suppression: why overlapping boxes at the same spot are merged
  - IoU (Intersection over Union): the standard metric for box quality
  - How YOLO/SSD replace this with a single forward pass

How to run:
  python CNN\object_detection_simple.py   (from PYTHON\ folder)

Estimated RAM: ~200MB | Time: ~60s (sliding window over multiple test images)
Model: 2-conv binary CNN (~30k params). Synthetic images. No download.
Output: CNN\outputs\object_detection.png
"""

import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

os.makedirs("CNN/outputs", exist_ok=True)
rng = np.random.RandomState(42)
torch.manual_seed(42)

IMG_SIZE  = 128
CROP_SIZE = 32   # patch classifier window
LABELS    = ["background", "shape"]

# ─── SCENE & PATCH GENERATORS ─────────────────────────────────────────────────

def draw_shape(img, colour=None):
    """Draw one random shape (circle or square) on img. Returns bounding box."""
    if colour is None:
        colour = tuple(int(c) for c in rng.randint(30, 180, 3))
    shape_type = rng.randint(0, 2)
    # Ensure the shape fits inside the image with a margin
    margin = CROP_SIZE // 2
    cx = rng.randint(margin, IMG_SIZE - margin)
    cy = rng.randint(margin, IMG_SIZE - margin)
    radius = rng.randint(12, 22)
    if shape_type == 0:   # circle
        cv2.circle(img, (cx, cy), radius, colour, -1)
    else:                 # square
        cv2.rectangle(img, (cx - radius, cy - radius),
                      (cx + radius, cy + radius), colour, -1)
    x1 = max(0, cx - radius - 2)
    y1 = max(0, cy - radius - 2)
    x2 = min(IMG_SIZE - 1, cx + radius + 2)
    y2 = min(IMG_SIZE - 1, cy + radius + 2)
    return (x1, y1, x2, y2)


def make_scene(n_shapes=None):
    """Create a 128x128 scene with 1-3 shapes."""
    img = np.full((IMG_SIZE, IMG_SIZE, 3), 240, dtype=np.uint8)  # light grey bg
    if n_shapes is None:
        n_shapes = rng.randint(1, 4)
    boxes = []
    for _ in range(n_shapes):
        box = draw_shape(img)
        boxes.append(box)
    return img, boxes


def crop_patch(img, cx, cy, size=CROP_SIZE):
    """Extract a crop centred at (cx, cy), padded at borders."""
    half = size // 2
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(IMG_SIZE, cx + half), min(IMG_SIZE, cy + half)
    patch = img[y1:y2, x1:x2]
    if patch.shape[0] != size or patch.shape[1] != size:
        patch = cv2.resize(patch, (size, size))
    return patch


def patch_contains_shape(patch_xyxy, gt_boxes, iou_thresh=0.3):
    """Return True if patch overlaps any GT box with IoU > threshold."""
    px1, py1, px2, py2 = patch_xyxy
    for bx1, by1, bx2, by2 in gt_boxes:
        ix1 = max(px1, bx1); iy1 = max(py1, by1)
        ix2 = min(px2, bx2); iy2 = min(py2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            continue
        area_p = (px2 - px1) * (py2 - py1)
        area_b = (bx2 - bx1) * (by2 - by1)
        iou    = inter / (area_p + area_b - inter + 1e-6)
        if iou > iou_thresh:
            return True
    return False


def build_patch_dataset(n_scenes=200):
    """Sample positive (shape) and negative (background) patches."""
    pos_patches, neg_patches = [], []
    stride = CROP_SIZE // 2
    for _ in range(n_scenes):
        scene, gt_boxes = make_scene(n_shapes=rng.randint(1, 3))
        for cy in range(CROP_SIZE//2, IMG_SIZE - CROP_SIZE//2, stride):
            for cx in range(CROP_SIZE//2, IMG_SIZE - CROP_SIZE//2, stride):
                patch_box = (cx - CROP_SIZE//2, cy - CROP_SIZE//2,
                             cx + CROP_SIZE//2, cy + CROP_SIZE//2)
                patch = crop_patch(scene, cx, cy)
                patch_f = patch.astype(np.float32) / 255.0
                if patch_contains_shape(patch_box, gt_boxes):
                    pos_patches.append(patch_f)
                else:
                    neg_patches.append(patch_f)

    # Balance: keep equal numbers of pos/neg
    n_each = min(len(pos_patches), len(neg_patches), 2000)
    rng.shuffle(pos_patches); rng.shuffle(neg_patches)
    patches = pos_patches[:n_each] + neg_patches[:n_each]
    labels  = [1] * n_each + [0] * n_each
    patches = np.array(patches).transpose(0, 3, 1, 2)
    labels  = np.array(labels, dtype=np.int64)
    idx     = rng.permutation(len(labels))
    return patches[idx], labels[idx]

# ─── PATCH CLASSIFIER MODEL ───────────────────────────────────────────────────

class PatchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8x8
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )
    def forward(self, x): return self.net(x)

# ─── NMS ──────────────────────────────────────────────────────────────────────

def nms(boxes_scores, iou_thresh=0.3):
    """Non-Maximum Suppression: remove overlapping detections."""
    if not boxes_scores:
        return []
    boxes_scores = sorted(boxes_scores, key=lambda x: -x[4])  # sort by score
    kept = []
    while boxes_scores:
        best = boxes_scores.pop(0)
        kept.append(best)
        bx1, by1, bx2, by2, _ = best
        remaining = []
        for bs in boxes_scores:
            x1, y1, x2, y2, _ = bs
            ix1 = max(bx1, x1); iy1 = max(by1, y1)
            ix2 = min(bx2, x2); iy2 = min(by2, y2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area1 = (bx2 - bx1) * (by2 - by1)
            area2 = (x2 - x1) * (y2 - y1)
            iou   = inter / (area1 + area2 - inter + 1e-6)
            if iou <= iou_thresh:
                remaining.append(bs)
        boxes_scores = remaining
    return kept

# ─── SLIDING WINDOW DETECTOR ──────────────────────────────────────────────────

def detect(model, scene_img, conf_thresh=0.85, stride=8):
    """Run sliding window over scene, return NMS-filtered detections."""
    model.eval()
    detections = []
    for cy in range(CROP_SIZE//2, IMG_SIZE - CROP_SIZE//2, stride):
        for cx in range(CROP_SIZE//2, IMG_SIZE - CROP_SIZE//2, stride):
            patch = crop_patch(scene_img, cx, cy)
            t = torch.tensor(patch.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
            with torch.no_grad():
                prob = torch.softmax(model(t), 1)[0, 1].item()
            if prob > conf_thresh:
                half = CROP_SIZE // 2
                detections.append((cx - half, cy - half, cx + half, cy + half, prob))
    return nms(detections, iou_thresh=0.3)

# ─── TRAINING ─────────────────────────────────────────────────────────────────

print("Simple Object Detector (Sliding Window CNN)")
print("=" * 55)
print("Building patch training set (positive/negative crops)...")
t0 = time.time()
X, y = build_patch_dataset(n_scenes=150)
print(f"  Patches: {len(X)} total ({(y==1).sum()} pos, {(y==0).sum()} neg)")

dl = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                batch_size=64, shuffle=True)

model     = PatchCNN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("Training patch classifier (3 epochs)...")
for epoch in range(3):
    model.train()
    tot_loss, correct = 0.0, 0
    for xb, yb in dl:
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * len(yb)
        correct  += (out.argmax(1) == yb).sum().item()
    acc = correct / len(X)
    print(f"  Epoch {epoch+1}/3  loss={tot_loss/len(X):.4f}  acc={acc:.3f}")
print(f"  Patch classifier trained in {time.time()-t0:.1f}s")

# ─── DETECTION ON TEST SCENES ─────────────────────────────────────────────────

print("\nRunning sliding-window detector on 4 test scenes...")
rng.seed(2025)
test_scenes, test_boxes = zip(*[make_scene(n_shapes=rng.randint(1, 4)) for _ in range(4)])

fig, axes = plt.subplots(2, 4, figsize=(15, 7))
fig.suptitle(
    "Simple Sliding-Window Object Detector\n"
    "Left cols: Ground Truth  |  Right cols: CNN Predictions (after NMS)\n"
    "Blue=GT box  Green=detected (conf>0.85)  Red=missed",
    fontsize=9
)

for i, (scene, gt_boxes) in enumerate(zip(test_scenes, test_boxes)):
    td = time.time()
    detected = detect(model, scene, conf_thresh=0.85, stride=8)
    det_time = time.time() - td

    # GT column
    ax_gt  = axes[0][i]
    gt_vis = scene.copy()
    for (x1, y1, x2, y2) in gt_boxes:
        cv2.rectangle(gt_vis, (x1, y1), (x2, y2), (0, 80, 200), 2)
    ax_gt.imshow(gt_vis)
    ax_gt.set_title(f"GT: {len(gt_boxes)} shape(s)", fontsize=8)
    ax_gt.axis("off")

    # Pred column
    ax_pr  = axes[1][i]
    pr_vis = scene.copy()
    for (x1, y1, x2, y2, conf) in detected:
        cv2.rectangle(pr_vis, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(pr_vis, f"{conf:.2f}", (x1, max(y1-3, 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 150, 0), 1)
    ax_pr.imshow(pr_vis)
    ax_pr.set_title(f"Det: {len(detected)}  ({det_time:.1f}s/frame)", fontsize=8)
    ax_pr.axis("off")

patch_count = sum(((IMG_SIZE - CROP_SIZE) // 8 + 1)**2 for _ in range(4))
print(f"  Sliding window inspects ~{(IMG_SIZE//8)**2} positions per frame")
print(f"  NMS reduces overlapping detections to clean bounding boxes")
print(f"  YOLO/SSD replace this with a single forward pass for ~100x speedup")

plt.tight_layout()
plt.savefig("CNN/outputs/object_detection.png", dpi=90)
plt.close()
print("Plot saved -> CNN/outputs/object_detection.png")
print("[DONE] object_detection_simple.py complete")
