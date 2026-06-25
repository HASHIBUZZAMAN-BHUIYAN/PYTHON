"""
Project: Image Autoencoder on Tiny 8×8 Grayscale Patterns
Teaches: encoding visual data, reconstruction quality, latent space interpolation.
~50 MB RAM, ~5s on CPU
"""
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42); np.random.seed(42)

# ─── Generate 300 synthetic 8×8 images: circles, horizontal bars, vertical bars
def make_images(n_each=100):
    images, labels = [], []
    for _ in range(n_each):
        # Type 0: circle pattern
        img = np.zeros((8,8))
        cx,cy=3.5,3.5; r=np.random.uniform(1.5,3.0)
        for i in range(8):
            for j in range(8):
                if abs(np.sqrt((i-cx)**2+(j-cy)**2)-r)<0.8: img[i,j]=1.0
        images.append(img.flatten()); labels.append(0)
    for _ in range(n_each):
        # Type 1: horizontal bars
        img=np.zeros((8,8))
        for r in np.random.choice(8, np.random.randint(2,5), replace=False): img[r,:]=1.0
        images.append(img.flatten()); labels.append(1)
    for _ in range(n_each):
        # Type 2: vertical bars
        img=np.zeros((8,8))
        for c in np.random.choice(8, np.random.randint(2,5), replace=False): img[:,c]=1.0
        images.append(img.flatten()); labels.append(2)
    idx = np.random.permutation(len(images))
    return (np.array(images)[idx].astype(np.float32),
            np.array(labels)[idx])

X_np, y = make_images(100)
X = torch.tensor(X_np)

class ImageAE(nn.Module):
    def __init__(self, latent=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32,16), nn.ReLU(),
            nn.Linear(16, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent,16), nn.ReLU(),
            nn.Linear(16,32),     nn.ReLU(),
            nn.Linear(32,64),     nn.Sigmoid(),
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

ae  = ImageAE(latent=4)
opt = optim.Adam(ae.parameters(), lr=1e-3)
crit= nn.MSELoss()

losses = []
for epoch in range(400):
    ae.train()
    Xh, _ = ae(X)
    loss = crit(Xh, X)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())

ae.eval()
with torch.no_grad():
    X_hat, Z = ae(X)
recon_err = ((X - X_hat)**2).mean(dim=1).numpy()

print(f"=== Image Autoencoder (64D → 4D → 64D) ===")
print(f"Final loss: {losses[-1]:.5f}")
print(f"Mean reconstruction error: {recon_err.mean():.4f}")

# ─── Visualize: original vs reconstructed ────────────────────────────────────
fig, axes = plt.subplots(4, 6, figsize=(10, 7))
for i in range(6):
    orig  = X[i].numpy().reshape(8,8)
    recon = X_hat[i].numpy().reshape(8,8)
    axes[0,i].imshow(orig, cmap="gray", vmin=0, vmax=1)
    axes[1,i].imshow(recon, cmap="gray", vmin=0, vmax=1)
    axes[0,i].axis("off"); axes[1,i].axis("off")
    axes[0,i].set_title(f"Type {y[i]}", fontsize=8)
    axes[1,i].set_title(f"err={recon_err[i]:.3f}", fontsize=8)

axes[0,0].set_ylabel("Original", fontsize=9); axes[1,0].set_ylabel("Recon.", fontsize=9)

# Loss curve
axes[2,0].plot(losses, color="steelblue")
for ax in axes[2,1:]: ax.axis("off")
axes[2,0].set_title("Loss"); axes[2,0].set_xlabel("Epoch")

# Latent space scatter
Z_np = Z.numpy()
sc = axes[3,0].scatter(Z_np[:,0], Z_np[:,1], c=y, cmap="tab10", s=15, alpha=0.7)
axes[3,0].set_title("Latent space (z0 vs z1)"); axes[3,0].set_xlabel("z0"); axes[3,0].set_ylabel("z1")
plt.colorbar(sc, ax=axes[3,0])
for ax in axes[3,1:]: ax.axis("off")

plt.suptitle("Image Autoencoder: Original vs Reconstructed", fontsize=10)
plt.tight_layout(); plt.savefig("image_ae.png", dpi=85); plt.close()
print("Saved image_ae.png")
