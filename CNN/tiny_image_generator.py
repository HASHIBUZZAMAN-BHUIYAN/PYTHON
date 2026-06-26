"""
Tiny Convolutional Autoencoder / Image Generator
==================================================
What it does:
  Trains a very small convolutional autoencoder on 28x28 synthetic shape
  images (circles, squares, triangles drawn in-code). After training:
    1. Shows reconstruction quality (input vs reconstructed images)
    2. Samples points in the 8-dimensional latent space via interpolation
       between two encoded images, generating a smooth "morphing" sequence
    3. Shows a random sampling grid by drawing random latent vectors
    4. Plots the 2D t-SNE projection of latent vectors coloured by class,
       showing that the encoder has learned a structured latent space

  This is the same architecture as:
    - Autoencoders used for anomaly detection (DL/anomaly_detection_realworld.py)
    - The encoder half of a Variational Autoencoder (VAE)
    - The first generation of image generators (before GANs/diffusion)

What it teaches:
  - Encoder-decoder architecture: compress to latent -> reconstruct
  - Why the latent space captures meaningful structure
    (shapes from same class cluster together in 2D t-SNE)
  - Linear interpolation in latent space: halfway between "circle" and
    "square" in latent space produces something in between visually
  - Why tiny models matter: the whole pipeline trains in <30s on CPU

How to run:
  python CNN\tiny_image_generator.py   (from PYTHON\ folder)

Estimated RAM: ~200MB | Time: ~30s (1000 samples x 15 epochs)
Model: Conv autoencoder, ~25k params. 28x28 synthetic shapes. No download.
Output: CNN\outputs\tiny_generator.png
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

SIZE = 28     # keep tiny for CPU speed
LATENT = 8   # latent space dimensionality
LABELS = ["circle", "square", "triangle"]

# ─── SYNTHETIC SHAPE DATASET ──────────────────────────────────────────────────

def make_circle(size=SIZE):
    img = np.zeros((size, size), dtype=np.uint8)
    cx, cy = rng.randint(8, size-8, 2)
    r = rng.randint(5, 10)
    cv2.circle(img, (int(cx), int(cy)), int(r), 255, -1)
    return img

def make_square(size=SIZE):
    img = np.zeros((size, size), dtype=np.uint8)
    x1 = rng.randint(4, size//2 - 2)
    y1 = rng.randint(4, size//2 - 2)
    side = rng.randint(6, 12)
    cv2.rectangle(img, (x1, y1), (x1 + side, y1 + side), 255, -1)
    return img

def make_triangle(size=SIZE):
    img = np.zeros((size, size), dtype=np.uint8)
    cx, cy = rng.randint(8, size-8), rng.randint(8, size-8)
    r = rng.randint(6, 11)
    pts = np.array([
        [cx, cy - r],
        [cx - int(r * 0.87), cy + r // 2],
        [cx + int(r * 0.87), cy + r // 2],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], 255)
    return img

SHAPE_FNS = [make_circle, make_square, make_triangle]

def build_dataset(n_per_class=334):
    images, labels = [], []
    for cls_idx, fn in enumerate(SHAPE_FNS):
        for _ in range(n_per_class):
            img  = fn()
            img  = np.clip(img.astype(int) + rng.randint(-15, 16, img.shape), 0, 255).astype(np.uint8)
            images.append(img)
            labels.append(cls_idx)
    images = np.array(images, dtype=np.float32) / 255.0
    images = images[:, np.newaxis, :, :]     # add channel dim: N,1,H,W
    labels = np.array(labels, dtype=np.int64)
    idx    = rng.permutation(len(labels))
    return images[idx], labels[idx]

# ─── MODEL ────────────────────────────────────────────────────────────────────

class TinyAutoEncoder(nn.Module):
    def __init__(self, latent_dim=LATENT):
        super().__init__()
        # Encoder: 28x28 -> 14x14 -> 7x7 -> latent
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14x14
            nn.Conv2d(16, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 7x7
            nn.Flatten(),
            nn.Linear(8 * 7 * 7, latent_dim),
        )
        # Decoder: latent -> 7x7 -> 14x14 -> 28x28
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8 * 7 * 7), nn.ReLU(),
            nn.Unflatten(1, (8, 7, 7)),
            nn.ConvTranspose2d(8, 16, 2, stride=2),  nn.ReLU(),   # 14x14
            nn.ConvTranspose2d(16, 1,  2, stride=2),  nn.Sigmoid(), # 28x28
        )

    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)
    def forward(self, x): return self.decode(self.encode(x))

# ─── TRAINING ─────────────────────────────────────────────────────────────────

print("Tiny Convolutional Autoencoder / Image Generator")
print("=" * 55)
print("Generating 1002 synthetic shape images (28x28)...")

X, y = build_dataset(n_per_class=334)
dl   = DataLoader(TensorDataset(torch.tensor(X)), batch_size=64, shuffle=True)

model     = TinyAutoEncoder(latent_dim=LATENT)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
t0 = time.time()
EPOCHS = 15
losses = []
for epoch in range(EPOCHS):
    model.train()
    tot_loss = 0.0
    for (xb,) in dl:
        optimizer.zero_grad()
        recon = model(xb)
        loss  = criterion(recon, xb)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * len(xb)
    avg_loss = tot_loss / len(X)
    losses.append(avg_loss)
    if (epoch + 1) % 3 == 0:
        print(f"  Epoch {epoch+1:>2}/{EPOCHS}  recon_loss={avg_loss:.5f}")
print(f"  Trained in {time.time()-t0:.1f}s")

# ─── ENCODE ALL DATA (for t-SNE and latent inspection) ───────────────────────
model.eval()
X_t = torch.tensor(X)
with torch.no_grad():
    Z_all  = model.encode(X_t).numpy()    # (N, LATENT)
    X_recon = model(X_t).numpy()          # (N, 1, 28, 28)

# ─── LATENT INTERPOLATION (circle -> square morphing) ────────────────────────
circle_idx   = np.where(y == 0)[0][0]
square_idx   = np.where(y == 1)[0][0]
triangle_idx = np.where(y == 2)[0][0]

z_circle   = Z_all[circle_idx]
z_square   = Z_all[square_idx]
z_triangle = Z_all[triangle_idx]

N_INTERP = 7
interp_imgs = []
for i in range(N_INTERP):
    t = i / (N_INTERP - 1)
    z = (1 - t) * z_circle + t * z_square   # linear interpolation
    with torch.no_grad():
        img = model.decode(torch.tensor(z).unsqueeze(0)).numpy()[0, 0]
    interp_imgs.append(img)

# ─── RANDOM SAMPLES FROM LATENT SPACE ────────────────────────────────────────
z_mean = Z_all.mean(axis=0)
z_std  = Z_all.std(axis=0)
sampled_imgs = []
for _ in range(12):
    z = rng.normal(z_mean, z_std).astype(np.float32)
    with torch.no_grad():
        img = model.decode(torch.tensor(z).unsqueeze(0)).numpy()[0, 0]
    sampled_imgs.append(img)

# ─── T-SNE LATENT PROJECTION ─────────────────────────────────────────────────
from sklearn.manifold import TSNE
tsne  = TSNE(n_components=2, random_state=42, perplexity=30)
Z_2d  = tsne.fit_transform(Z_all)

# ─── PLOT ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = fig.add_gridspec(3, 10, hspace=0.45, wspace=0.15)

# Row 0: reconstruction comparison (3 inputs, 3 reconstructions)
row0_title = fig.add_subplot(gs[0, :])
row0_title.axis("off")
row0_title.set_title("Row 1: Original vs Reconstructed  |  Row 2: Latent Interpolation (circle->square)  |  Row 3: t-SNE + Random Samples",
                     fontsize=9)
for i, (cls_idx, label) in enumerate([(0, "circle"), (1, "square"), (2, "triangle")]):
    idx = np.where(y == cls_idx)[0][i % 5]
    ax_in  = fig.add_subplot(gs[0, i*2])
    ax_out = fig.add_subplot(gs[0, i*2 + 1])
    ax_in.imshow(X[idx, 0], cmap="gray", vmin=0, vmax=1)
    ax_in.set_title(f"Input\n({label})", fontsize=7)
    ax_in.axis("off")
    ax_out.imshow(X_recon[idx, 0], cmap="gray", vmin=0, vmax=1)
    ax_out.set_title(f"Recon\n({label})", fontsize=7)
    ax_out.axis("off")

# Loss curve (right side row 0)
ax_loss = fig.add_subplot(gs[0, 7:10])
ax_loss.plot(range(1, EPOCHS+1), losses, "o-", color="steelblue", markersize=3)
ax_loss.set_title("Reconstruction Loss", fontsize=8)
ax_loss.set_xlabel("Epoch"); ax_loss.grid(alpha=0.3)

# Row 1: interpolation sequence
for i, img in enumerate(interp_imgs):
    ax = fig.add_subplot(gs[1, i])
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"t={i/(N_INTERP-1):.2f}", fontsize=7)
    ax.axis("off")
# Label
ax_interp_lbl = fig.add_subplot(gs[1, 7:10])
ax_interp_lbl.axis("off")
ax_interp_lbl.text(0.1, 0.5,
    "Latent space interpolation:\n"
    "circle (t=0) -> square (t=1)\n\n"
    "The encoder has learned a\n"
    "smooth, meaningful latent\n"
    "space -- halfway between\n"
    "the two classes is a shape\n"
    "that looks like both.",
    transform=ax_interp_lbl.transAxes, fontsize=8, va="center",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

# Row 2: t-SNE (left) + random samples (right 6)
ax_tsne = fig.add_subplot(gs[2, 0:4])
COLOURS = ["#1565C0", "#B71C1C", "#2E7D32"]
for cls_idx, (label, col) in enumerate(zip(LABELS, COLOURS)):
    mask = y == cls_idx
    ax_tsne.scatter(Z_2d[mask, 0], Z_2d[mask, 1], s=5, c=col, label=label, alpha=0.6)
ax_tsne.set_title("t-SNE of Latent Space\n(coloured by class)", fontsize=8)
ax_tsne.legend(fontsize=7, markerscale=3)
ax_tsne.axis("off")

for i, img in enumerate(sampled_imgs[:6]):
    ax = fig.add_subplot(gs[2, 4 + i])
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Sample {i+1}", fontsize=7)
    ax.axis("off")

plt.suptitle("Tiny Conv Autoencoder: 28x28 Shapes  |  8-dim Latent Space  |  No download needed", fontsize=10)
plt.savefig("CNN/outputs/tiny_generator.png", dpi=90)
plt.close()
print("Plot saved -> CNN/outputs/tiny_generator.png")
print("[DONE] tiny_image_generator.py complete")
