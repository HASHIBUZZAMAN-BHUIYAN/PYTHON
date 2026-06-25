# Advanced Day 31 — Autoencoders
# ~80 MB RAM, ~10s on CPU

print("""
=== Autoencoders — Day 31 ===

An Autoencoder has two parts:
  Encoder: X → latent code z  (compress)
  Decoder: z → X_hat          (reconstruct)

Trained to minimize: MSE(X, X_hat)

Uses:
  - Dimensionality reduction (like PCA but nonlinear)
  - Denoising: train with noisy input, reconstruct clean
  - Anomaly detection: high reconstruction error = anomaly
  - Generative modeling (VAE extension)
""")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# ─── SYNTHETIC DATA: 2D swiss roll projected to 8D ───────────────────────────
def make_data(n=300):
    t = np.linspace(0, 4*np.pi, n)
    x = t * np.cos(t); y = t * np.sin(t)
    xy = np.column_stack([x, y])
    xy = (xy - xy.mean(0)) / (xy.std(0) + 1e-8)
    proj = np.random.randn(2, 8)
    X8 = xy @ proj + 0.1*np.random.randn(n, 8)
    return X8.astype(np.float32), xy

X_np, xy = make_data(300)
X = torch.tensor(X_np)

# ─── AUTOENCODER ─────────────────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, input_dim),
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# ─── TRAINING ────────────────────────────────────────────────────────────────
print("=== Training Autoencoder (8D → 2D → 8D) ===")
ae   = Autoencoder(input_dim=8, latent_dim=2)
opt  = optim.Adam(ae.parameters(), lr=1e-3)
crit = nn.MSELoss()
losses = []
for epoch in range(300):
    ae.train()
    X_hat, z = ae(X)
    loss = crit(X_hat, X)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
    if (epoch+1) % 100 == 0:
        print(f"  Epoch {epoch+1:3d}  loss={loss.item():.5f}")

ae.eval()
with torch.no_grad():
    X_hat, Z = ae(X)
recon_err = ((X - X_hat)**2).mean(dim=1).numpy()
print(f"\n  Mean reconstruction error: {recon_err.mean():.5f}")
print(f"  Max reconstruction error:  {recon_err.max():.5f}")

# ─── DENOISING AUTOENCODER ───────────────────────────────────────────────────
print("\n=== Denoising Autoencoder ===")
dae = Autoencoder(input_dim=8, latent_dim=2)
opt2= optim.Adam(dae.parameters(), lr=1e-3)

for epoch in range(300):
    noise_level = 0.5
    X_noisy = X + noise_level * torch.randn_like(X)
    X_hat_d, _ = dae(X_noisy)
    loss = crit(X_hat_d, X)  # reconstruct CLEAN from NOISY
    opt2.zero_grad(); loss.backward(); opt2.step()
    if (epoch+1) % 100 == 0:
        print(f"  Epoch {epoch+1:3d}  loss={loss.item():.5f}")

dae.eval()
X_noisy_test = X + 0.5*torch.randn_like(X)
with torch.no_grad():
    X_denoised, _ = dae(X_noisy_test)
noisy_err    = ((X_noisy_test - X)**2).mean(dim=1).numpy().mean()
denoised_err = ((X_denoised   - X)**2).mean(dim=1).numpy().mean()
print(f"  Noisy MSE:    {noisy_err:.4f}")
print(f"  Denoised MSE: {denoised_err:.4f}")
print(f"  Denoising improvement: {(noisy_err-denoised_err)/noisy_err:.1%}")

# ─── ANOMALY DETECTION ───────────────────────────────────────────────────────
print("\n=== Anomaly Detection ===")
X_normal  = X[:250]
X_anomaly = 3.0 + 0.3*torch.randn(10, 8)  # out-of-distribution

ae2 = Autoencoder(input_dim=8, latent_dim=2)
opt3= optim.Adam(ae2.parameters(), lr=1e-3)
for _ in range(500):
    Xh, _ = ae2(X_normal)
    loss = crit(Xh, X_normal)
    opt3.zero_grad(); loss.backward(); opt3.step()

ae2.eval()
with torch.no_grad():
    norm_err = ((ae2(X_normal)[0] - X_normal)**2).mean(dim=1).numpy()
    anom_err = ((ae2(X_anomaly)[0] - X_anomaly)**2).mean(dim=1).numpy()

threshold = np.percentile(norm_err, 95)
print(f"  Normal reconstruction error  (mean): {norm_err.mean():.4f}")
print(f"  Anomaly reconstruction error (mean): {anom_err.mean():.4f}")
print(f"  Threshold (95th percentile of normal): {threshold:.4f}")
print(f"  Anomalies detected: {(anom_err > threshold).sum()}/{len(anom_err)}")

# ─── VISUALIZE ────────────────────────────────────────────────────────────────
Z_np = Z.detach().numpy()
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].plot(losses, color="steelblue"); axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch")
axes[1].scatter(Z_np[:,0], Z_np[:,1], c=np.arange(len(Z_np)), cmap="viridis", s=20, alpha=0.7)
axes[1].set_title("Latent Space (2D)"); axes[1].set_xlabel("z1"); axes[1].set_ylabel("z2")
all_err = np.concatenate([norm_err, anom_err])
colors  = ["green"]*len(norm_err) + ["red"]*len(anom_err)
axes[2].scatter(range(len(all_err)), all_err, c=colors, s=30, alpha=0.7)
axes[2].axhline(threshold, color="k", linestyle="--", label=f"threshold={threshold:.3f}")
axes[2].set_title("Reconstruction Error"); axes[2].set_xlabel("Sample"); axes[2].legend()
plt.suptitle("Autoencoder Day 31", fontsize=11)
plt.tight_layout(); plt.savefig("ae_lesson.png", dpi=85); plt.close()
print("\nSaved ae_lesson.png")
