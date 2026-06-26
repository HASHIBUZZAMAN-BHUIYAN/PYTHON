"""
Image Filters Playground (Convolution Kernels)
================================================
What it does:
  Applies 7 classical convolution kernels to a procedurally generated test
  image and shows the results in a single grid figure with mathematical
  explanations for each kernel.

  Kernels demonstrated:
    1. Gaussian blur (5x5)   -- low-pass filter: smooths noise
    2. Sharpen               -- high-pass: enhances edges by subtracting blur
    3. Sobel X               -- horizontal gradient: detects vertical edges
    4. Sobel Y               -- vertical gradient: detects horizontal edges
    5. Laplacian             -- second derivative: detects all edges
    6. Emboss                -- directional shadow effect
    7. Custom "glow" kernel  -- weighted average with centre boost

  Under each filter result, prints the actual kernel matrix (values shown).

  WHY THIS MATTERS FOR CNNs:
    In a CNN's first layer, the model LEARNS these kernels from data.
    When you look at visualised CNN first-layer filters (see
    cnn_feature_map_visualizer.py), you'll recognise Gabor-like edge
    detectors, colour blobs, and frequency-selective patterns -- the
    same family of operations these manual kernels implement, but
    optimised for the specific task by gradient descent.

What it teaches:
  - What a convolution kernel is: a small weight matrix that slides
    across the image computing a weighted sum at each position
  - How different kernels extract different features:
    Sobel -> gradient/edge | Gaussian -> smoothing | Laplacian -> curvature
  - Why CNN first layers always learn edge-like detectors:
    edges are the most information-dense feature in natural images
  - Separable kernels: Gaussian is separable (Gx * Gy) -- why this
    makes large blurs computationally efficient

How to run:
  python CNN\image_filters_playground.py   (from PYTHON\ folder)

Estimated RAM: <50MB | Time: <2s
Model: pure OpenCV convolutions, no neural network. No download.
Output: CNN\outputs\image_filters.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

os.makedirs("CNN/outputs", exist_ok=True)
rng = np.random.RandomState(42)

# ─── TEST IMAGE ───────────────────────────────────────────────────────────────

def make_test_image(h=256, w=256):
    """Geometric scene with clear edges and textures for filter demonstration."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 200
    for col in range(w):
        img[:, col, :] = 180 + int(40 * col / w)
    cv2.rectangle(img, (20, 20),   (100, 100), (60, 60, 180), -1)
    cv2.circle(img,    (180, 60),  45,          (200, 80, 40), -1)
    cv2.rectangle(img, (50, 140),  (150, 220),  (60, 150, 60), -1)
    cv2.ellipse(img,   (200, 180), (50, 30), 30, 0, 360, (160, 30, 160), -1)
    for y in range(30, 100, 8):
        cv2.line(img, (20, y), (100, y), (255, 255, 255), 1)
    for d in range(0, h, 20):
        cv2.line(img, (d, 0), (0, d), (100, 100, 100), 1)
    img[180:230, 180:230] = rng.randint(80, 220, (50, 50, 3), dtype=np.uint8)
    return img

# ─── KERNEL DEFINITIONS ───────────────────────────────────────────────────────

# Gaussian 5x5 (sigma~1.5)
GAUSSIAN = np.array([
    [1,  4,  7,  4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1,  4,  7,  4, 1],
], dtype=np.float32) / 273.0

SHARPEN = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0],
], dtype=np.float32)

SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
], dtype=np.float32)

SOBEL_Y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1],
], dtype=np.float32)

LAPLACIAN = np.array([
    [ 0, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  0],
], dtype=np.float32)

EMBOSS = np.array([
    [-2, -1, 0],
    [-1,  1, 1],
    [ 0,  1, 2],
], dtype=np.float32)

GLOW = np.array([
    [0.05, 0.10, 0.05],
    [0.10, 0.40, 0.10],
    [0.05, 0.10, 0.05],
], dtype=np.float32)

def apply_kernel(img_bgr, kernel, mode="colour", clip=True):
    """
    Apply a convolution kernel to the image.
    mode='colour': apply per channel
    mode='grey': convert to grey first, return greyscale result
    mode='abs': apply per channel, take absolute value (for gradient kernels)
    """
    if mode == "grey":
        grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        result = cv2.filter2D(grey, -1, kernel)
        if clip:
            result = np.clip(result, 0, 255).astype(np.uint8)
            return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        return result
    elif mode == "abs":
        grey   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        result = np.abs(cv2.filter2D(grey, -1, kernel))
        result = np.clip(result * 3, 0, 255).astype(np.uint8)  # scale for visibility
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    else:
        result = cv2.filter2D(img_bgr.astype(np.float32), -1, kernel)
        if clip:
            result = np.clip(result, 0, 255).astype(np.uint8)
        return result

def kernel_to_str(k, decimals=2):
    """Format a kernel as a readable multi-line string."""
    lines = []
    for row in k:
        lines.append("  " + "  ".join(f"{v:+.{decimals}f}" for v in row))
    return "\n".join(lines)

# ─── DEMO ─────────────────────────────────────────────────────────────────────

print("Image Filters Playground")
print("=" * 55)
print("Generating test image and applying 7 convolution kernels...")

img_bgr = make_test_image()

FILTERS = [
    ("Original",       img_bgr,                                            None,       "Reference image (no kernel)"),
    ("Gaussian Blur",  apply_kernel(img_bgr, GAUSSIAN, "colour"),          GAUSSIAN,   "Low-pass: removes high-freq noise"),
    ("Sharpen",        apply_kernel(img_bgr, SHARPEN,  "colour"),          SHARPEN,    "High-pass: enhances edges"),
    ("Sobel X",        apply_kernel(img_bgr, SOBEL_X,  "abs"),             SOBEL_X,    "Horizontal gradient -> vertical edges"),
    ("Sobel Y",        apply_kernel(img_bgr, SOBEL_Y,  "abs"),             SOBEL_Y,    "Vertical gradient -> horizontal edges"),
    ("Laplacian",      apply_kernel(img_bgr, LAPLACIAN, "abs"),            LAPLACIAN,  "2nd derivative -> all edge directions"),
    ("Emboss",         np.clip(apply_kernel(img_bgr, EMBOSS, "grey",
                                            clip=False).astype(float) + 128, 0, 255).astype(np.uint8)
                       if True else None,                                   EMBOSS,     "Directional shadow effect"),
    ("Glow Blend",     apply_kernel(img_bgr, GLOW, "colour"),              GLOW,       "Soft centre-weighted average"),
]
FILTERS[6] = (
    "Emboss",
    cv2.cvtColor(
        np.clip(
            cv2.filter2D(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), -1, EMBOSS) + 128,
            0, 255
        ).astype(np.uint8),
        cv2.COLOR_GRAY2BGR
    ),
    EMBOSS, "Directional shadow effect"
)

fig, axes = plt.subplots(2, 8, figsize=(20, 6),
                         gridspec_kw={"height_ratios": [3, 1]})
fig.suptitle(
    "Image Convolution Kernels Playground\n"
    "WHY IT MATTERS FOR CNNs: first-layer CNN filters LEARN these same operations from data",
    fontsize=10
)

for col, (title, result_bgr, kernel, note) in enumerate(FILTERS):
    ax_img  = axes[0][col]
    ax_kern = axes[1][col]

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    ax_img.imshow(result_rgb)
    ax_img.set_title(title, fontsize=8, fontweight="bold")
    ax_img.set_xlabel(note, fontsize=6.5, ha="center")
    ax_img.axis("off")

    ax_kern.axis("off")
    if kernel is not None and kernel.size <= 25:   # only show <=5x5
        krows, kcols = kernel.shape
        for i in range(krows):
            for j in range(kcols):
                v = kernel[i, j]
                fc = "red" if v < 0 else ("green" if v > 0 else "grey")
                ax_kern.text(j / kcols, 1 - (i + 0.7) / krows,
                             f"{v:+.2f}", ha="center", va="center",
                             fontsize=5.5, color=fc,
                             transform=ax_kern.transAxes)
        ax_kern.set_title(f"{krows}x{kcols} kernel", fontsize=6)
    elif kernel is None:
        ax_kern.text(0.5, 0.5, "No kernel\n(original)", ha="center", va="center",
                     fontsize=6.5, transform=ax_kern.transAxes)
    else:
        ax_kern.text(0.5, 0.5, f"{kernel.shape} kernel\n(too large to show)",
                     ha="center", va="center", fontsize=6.5,
                     transform=ax_kern.transAxes)

plt.tight_layout()
plt.savefig("CNN/outputs/image_filters.png", dpi=90)
plt.close()
print("Plot saved -> CNN/outputs/image_filters.png")
print()
print("KERNEL MEANINGS:")
print("  Gaussian (all positive, sums to 1): smoothing / low-pass")
print("  Sobel   (negative left, positive right): gradient = edge direction")
print("  Sharpen (large centre, negative neighbours): enhances local contrast")
print("  Laplacian (negative neighbours, centre=4): second derivative = edges")
print("  Emboss  (diagonal gradient): shadow-like 3D effect")
print()
print("CNN FIRST LAYER CONNECTION:")
print("  When you train a CNN on natural images and visualise its first-layer")
print("  filters, you see: Gabor-like edge detectors (Sobel family), colour")
print("  blobs, and frequency-selective patterns. The network LEARNS the")
print("  optimal versions of these kernels for its specific task.")
print("[DONE] image_filters_playground.py complete")
