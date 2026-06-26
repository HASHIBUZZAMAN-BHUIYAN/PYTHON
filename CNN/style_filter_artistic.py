"""
Artistic Style Filters (Classical CV)
========================================
What it does:
  Applies 4 lightweight OpenCV-based artistic filters to a procedurally
  generated test image (a colourful synthetic scene -- no download needed):
    1. Cartoon effect: bilateral filter (edge-preserving smoothing) +
       adaptive threshold edge overlay -> flat colours + crisp outlines
    2. Pencil sketch: grayscale inversion + Gaussian blur + divide trick
    3. Emboss: custom 3x3 emboss kernel convolution + level shift
    4. Oil painting approximation: median blur in LAB colour space on
       downsampled patches -> "painting-like" colour blobs

  Saves a before/after grid showing all 4 effects.

  *** IMPORTANT: This is NOT neural style transfer. ***
  True CNN-based neural style transfer (Gatys et al., 2015) extracts
  content features from a VGG-19 layer and style (Gram matrices) from
  another, then optimises a noise image to match both. That requires:
    - VGG-19 weights (~550MB download)
    - GPU or 5-20 minutes per image on CPU
  These classical filters produce SIMILAR visual results in milliseconds
  using only OpenCV convolutions -- ideal for real-time augmentation.

What it teaches:
  - How a CNN's first layers are literally learning these same operations
    (Gaussian, edge detection, colour manipulation) from data
  - Bilateral filter: why it's "edge-preserving" (smooths within regions
    but not across strong edges) -- key to the cartoon look
  - Convolution kernels: the emboss kernel IS a finite-difference gradient
    approximation, just like Sobel edges
  - Colour space manipulation: why LAB is better for perceptual operations
    than RGB (L=lightness decoupled from AB=colour)

How to run:
  python CNN\style_filter_artistic.py   (from PYTHON\ folder)

Estimated RAM: <100MB | Time: <3s
Model: pure OpenCV classical filters, no neural network. No download.
Output: CNN\outputs\style_filters.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

os.makedirs("CNN/outputs", exist_ok=True)
rng = np.random.RandomState(42)

# ─── GENERATE A COLOURFUL TEST IMAGE ─────────────────────────────────────────
# (so users don't need to provide their own image)

def make_test_image(h=256, w=256):
    """
    Procedurally generated scene: gradient sky, green hill, sun, coloured shapes.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky gradient (light blue top -> white bottom)
    for row in range(h // 2):
        t = row / (h // 2)
        img[row, :] = [int(220 + 35 * t), int(200 + 55 * t), int(120 + 135 * t)]  # BGR
    # Green hill
    for col in range(w):
        hill_y = int(h // 2 + 30 * np.sin(col * np.pi / w))
        img[hill_y:, col] = [30, 120, 40]  # dark green (BGR)
    # Yellow sun
    cv2.circle(img, (200, 50), 28, (30, 220, 240), -1)   # yellow in BGR
    cv2.circle(img, (200, 50), 30, (20, 200, 220), 2)
    # A few coloured shapes
    cv2.rectangle(img, (20, 100), (80, 160), (100, 50, 200), -1)  # purple box
    cv2.circle(img, (130, 140), 22, (200, 80, 30), -1)            # blue circle
    cv2.circle(img, (50, 190), 18, (40, 200, 200), -1)            # yellow circle
    # Add some noise
    noise = rng.randint(-15, 16, img.shape, dtype=np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

# ─── FILTERS ──────────────────────────────────────────────────────────────────

def cartoon_filter(img):
    """
    1. Bilateral filter multiple times -> smooth flat regions, keep edges
    2. Greyscale + adaptive threshold -> edge mask
    3. Multiply smoothed colour image by edge mask (darken edges)
    """
    smooth = img.copy()
    for _ in range(3):
        smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=75, sigmaSpace=75)
    grey   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges  = cv2.adaptiveThreshold(grey, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, blockSize=9, C=2)
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # Darken where edges exist (edges=0 -> black)
    cartoon = cv2.bitwise_and(smooth, edges_3ch)
    return cartoon


def pencil_sketch(img):
    """
    Greyscale inversion divide trick:
    sketch(x) = clip(grey / (1 - blurred/255))
    Produces a white paper + pencil stroke effect.
    """
    grey     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv      = 255 - grey
    blurred  = cv2.GaussianBlur(inv, (21, 21), 0)
    # Avoid division by zero
    denom    = 255 - blurred.astype(np.float32)
    denom    = np.where(denom == 0, 1, denom)
    sketch   = np.clip(grey.astype(np.float32) * 255.0 / denom, 0, 255).astype(np.uint8)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def emboss_filter(img):
    """
    Apply a 3x3 emboss convolution kernel then add 128 (grey level shift).
    Emboss kernel approximates a diagonal directional derivative.
    """
    kernel = np.array([[-2, -1, 0],
                       [-1,  1, 1],
                       [ 0,  1, 2]], dtype=np.float32)
    grey    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    emboss  = cv2.filter2D(grey, -1, kernel) + 128.0
    emboss  = np.clip(emboss, 0, 255).astype(np.uint8)
    return cv2.cvtColor(emboss, cv2.COLOR_GRAY2BGR)


def oil_painting_approx(img, radius=4, dynamic_ratio=0.5):
    """
    Simplified "oil painting": apply large median blur per channel in LAB
    then resize down+up to create painterly blobs.
    Not identical to Winnemöller's algorithm but achieves the visual effect.
    """
    lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Apply median blur separately per channel
    out  = lab.copy()
    for ch in range(3):
        out[:, :, ch] = cv2.medianBlur(lab[:, :, ch].astype(np.uint8), 7)
    # Downsample + upsample to exaggerate colour blobs
    h, w = img.shape[:2]
    small = cv2.resize(out.astype(np.uint8), (w // 2, h // 2))
    big   = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(big, cv2.COLOR_LAB2BGR)

# ─── DEMO ─────────────────────────────────────────────────────────────────────

print("Artistic Style Filters (Classical CV)")
print("=" * 55)
print("NOTE: These are OpenCV classical filters, NOT neural style transfer.")
print("Generating test image and applying 4 filters...")

img_bgr = make_test_image()
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

filters = [
    ("Original",      img_rgb),
    ("Cartoon",       cv2.cvtColor(cartoon_filter(img_bgr),      cv2.COLOR_BGR2RGB)),
    ("Pencil Sketch", cv2.cvtColor(pencil_sketch(img_bgr),        cv2.COLOR_BGR2RGB)),
    ("Emboss",        cv2.cvtColor(emboss_filter(img_bgr),        cv2.COLOR_BGR2RGB)),
    ("Oil Painting",  cv2.cvtColor(oil_painting_approx(img_bgr),  cv2.COLOR_BGR2RGB)),
]

fig, axes = plt.subplots(1, 5, figsize=(16, 4))
for ax, (title, fimg) in zip(axes, filters):
    ax.imshow(fimg)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")

# Annotations below images
notes = [
    "Procedurally\ngenerated scene\n(no download needed)",
    "Bilateral filter\n+adaptive threshold\nedge overlay",
    "Grey inversion\ndivide trick\n(pure arithmetic)",
    "3x3 diagonal\ngradient kernel\n+128 level shift",
    "LAB median blur\n+downsample/upsample\n(colour blob effect)",
]
for ax, note in zip(axes, notes):
    ax.set_xlabel(note, fontsize=7.5, ha="center")

fig.suptitle(
    "Classical CV Artistic Filters  [NOT neural style transfer]\n"
    "These are the same operations CNN first layers learn to do -- applied manually",
    fontsize=10
)
plt.tight_layout()
plt.savefig("CNN/outputs/style_filters.png", dpi=90)
plt.close()
print("Plot saved -> CNN/outputs/style_filters.png")
print()
print("CLASSICAL vs NEURAL STYLE TRANSFER:")
print("  Classical (this demo): OpenCV kernels, <3s, no download, deterministic")
print("  Neural style transfer: VGG-19 (~550MB), 5-20 min/image on CPU,")
print("    extracts Gram-matrix style features, optimises content+style loss")
print("  For real-time use (mobile AR filters): classical or distilled small nets")
print("[DONE] style_filter_artistic.py complete")
