# Advanced Day 32 — Tiny GANs
# ~60 MB RAM, ~10s on CPU

print("""
=== Generative Adversarial Networks (GANs) — Day 32 ===

GAN = two networks in competition:
  Generator G(z)    : takes noise z → generates fake samples
  Discriminator D(x): takes sample → outputs P(real)

Loss:
  D tries to maximize:  log D(x_real) + log(1 - D(G(z)))
  G tries to maximize:  log D(G(z))  (fool D)

Training:
  Alternate between updating D and G every batch.

Key pathologies:
  Mode collapse     — G generates only a few modes, ignoring diversity
  Training collapse — D becomes too strong; G's gradients vanish
""")

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42); np.random.seed(42)

# ─── TARGET DISTRIBUTION: mixture of 4 Gaussians in 2D ──────────────────────
def sample_real(n=256):
    means = [(2,2),(-2,2),(2,-2),(-2,-2)]
    data  = []
    for _ in range(n):
        mu = means[np.random.randint(4)]
        data.append(np.array(mu) + 0.4*np.random.randn(2))
    return torch.tensor(np.array(data, dtype=np.float32))

# ─── GENERATOR ───────────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, z_dim=8, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )
    def forward(self, z): return self.net(z)

# ─── DISCRIMINATOR ───────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)

Z_DIM = 8
G = Generator(z_dim=Z_DIM); D = Discriminator()
opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
bce = nn.BCELoss()

g_losses, d_losses = [], []
EPOCHS = 500; BATCH = 64

print("Training GAN on 2D mixture of Gaussians ...")
for epoch in range(EPOCHS):
    # ── Train Discriminator ──────────────────────────────────────────────────
    x_real = sample_real(BATCH)
    z      = torch.randn(BATCH, Z_DIM)
    x_fake = G(z).detach()
    loss_D = (bce(D(x_real), torch.ones(BATCH,1)) +
              bce(D(x_fake), torch.zeros(BATCH,1)))
    opt_D.zero_grad(); loss_D.backward(); opt_D.step()

    # ── Train Generator ───────────────────────────────────────────────────────
    z      = torch.randn(BATCH, Z_DIM)
    x_fake = G(z)
    loss_G = bce(D(x_fake), torch.ones(BATCH,1))
    opt_G.zero_grad(); loss_G.backward(); opt_G.step()

    g_losses.append(loss_G.item()); d_losses.append(loss_D.item())
    if (epoch+1) % 100 == 0:
        print(f"  Epoch {epoch+1:4d}  G={loss_G.item():.3f}  D={loss_D.item():.3f}")

# ─── SAMPLE FROM GENERATOR ───────────────────────────────────────────────────
G.eval()
with torch.no_grad():
    z_test  = torch.randn(512, Z_DIM)
    x_gen   = G(z_test).numpy()

x_real_vis = sample_real(512).numpy()

# ─── VISUALIZE ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].scatter(x_real_vis[:,0], x_real_vis[:,1], s=6, alpha=0.5, c="steelblue")
axes[0].set_title("Real Distribution"); axes[0].set_xlim(-5,5); axes[0].set_ylim(-5,5)
axes[1].scatter(x_gen[:,0], x_gen[:,1], s=6, alpha=0.5, c="tomato")
axes[1].set_title("Generated Distribution"); axes[1].set_xlim(-5,5); axes[1].set_ylim(-5,5)
axes[2].plot(g_losses, label="G loss", alpha=0.8, color="tomato")
axes[2].plot(d_losses, label="D loss", alpha=0.8, color="steelblue")
axes[2].set_title("Training Losses"); axes[2].set_xlabel("Step"); axes[2].legend()
plt.suptitle("2D GAN: Mixture of Gaussians", fontsize=11)
plt.tight_layout(); plt.savefig("gan_lesson.png", dpi=85); plt.close()
print("Saved gan_lesson.png")

print("""
=== Key Takeaways ===
  Nash equilibrium: D outputs 0.5 everywhere (can't tell real from fake)
  Monitor D loss: should stay ~0.69 (log 2); if → 0, D is winning
  Monitor G loss: should be decreasing; if stuck, mode collapse likely
  Learning rate trick: G usually needs 2× lower lr than D
""")
