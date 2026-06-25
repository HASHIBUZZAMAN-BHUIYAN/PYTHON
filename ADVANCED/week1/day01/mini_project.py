# Advanced Day 01 Mini-Project — NumPy Image Processing Basics
# Demonstrate array operations using a synthetic "image" (no actual file needed).
# ~30 MB RAM, <2s on CPU

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Create a synthetic grayscale "image" (64x64)
size = 64

# Base: gradient
x = np.linspace(0, 1, size)
y = np.linspace(0, 1, size)
xx, yy = np.meshgrid(x, y)
image = (xx + yy) / 2

# Add some shapes
cx, cy = size//2, size//2
for i in range(size):
    for j in range(size):
        if (i-cy)**2 + (j-cx)**2 < (size//4)**2:
            image[i,j] = 1.0   # white circle

# Add noise
noise = np.random.normal(0, 0.05, image.shape)
noisy = np.clip(image + noise, 0, 1)

# ─── OPERATIONS ──────────────────────────────────────────────────────────────
# Flip
flipped_h = noisy[:, ::-1]
flipped_v = noisy[::-1, :]

# Brighten / darken
bright = np.clip(noisy + 0.3, 0, 1)
dark   = np.clip(noisy - 0.3, 0, 1)

# Threshold (binary)
binary = (noisy > 0.5).astype(float)

# Horizontal blur (simple mean filter using slicing — no scipy)
blurred = (noisy[:-2, 1:-1] + noisy[1:-1, 1:-1] + noisy[2:, 1:-1]) / 3

print("=== Image Array Info ===")
print(f"Shape: {noisy.shape}, dtype: {noisy.dtype}")
print(f"Min: {noisy.min():.3f}, Max: {noisy.max():.3f}, Mean: {noisy.mean():.3f}")

# ─── VISUALISE ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
fig.suptitle("NumPy Image Operations")

pairs = [
    (noisy,     "Original (noisy)"),
    (bright,    "Brightened"),
    (dark,      "Darkened"),
    (binary,    "Threshold (>0.5)"),
    (flipped_h, "Horizontal flip"),
    (blurred,   "Simple blur"),
]
for ax, (img, title) in zip(axes.flat, pairs):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig("numpy_image_ops.png", dpi=80)
print("\nSaved numpy_image_ops.png")
plt.show()
