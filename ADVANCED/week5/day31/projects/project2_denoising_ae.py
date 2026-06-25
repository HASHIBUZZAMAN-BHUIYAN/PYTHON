"""
Project: Denoising Autoencoder on Sensor Signals
Teaches: corrupting 1D signals with noise, training to reconstruct clean versions,
         comparing noisy vs denoised vs original side-by-side.
~40 MB RAM, ~5s on CPU
"""
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42); np.random.seed(42)

# ─── Generate clean 1D sensor signals (length 32) ────────────────────────────
def make_signals(n=500, length=32):
    t = np.linspace(0, 2*np.pi, length)
    signals = []
    for i in range(n):
        freq   = np.random.uniform(0.5, 3.0)
        phase  = np.random.uniform(0, np.pi)
        amp    = np.random.uniform(0.5, 1.5)
        signal = amp * np.sin(freq * t + phase)
        signal+= 0.3 * amp * np.sin(2*freq * t)   # harmonic
        signals.append(signal)
    X = np.array(signals, dtype=np.float32)
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
    return X

X_clean = make_signals(500)
X_tensor= torch.tensor(X_clean)

class DenoisingAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32,16), nn.ReLU(),
            nn.Linear(16,8),  nn.ReLU(),
            nn.Linear(8, 4),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),  nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16,32),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

model = DenoisingAE()
opt   = optim.Adam(model.parameters(), lr=1e-3)
crit  = nn.MSELoss()

noise_std = 0.5
print(f"Training Denoising AE (noise_std={noise_std}) ...")
losses = []
for epoch in range(500):
    model.train()
    X_noisy = X_tensor + noise_std * torch.randn_like(X_tensor)
    X_hat   = model(X_noisy)
    loss    = crit(X_hat, X_tensor)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
    if (epoch+1) % 100 == 0:
        print(f"  Epoch {epoch+1}  loss={loss.item():.5f}")

model.eval()
X_noisy_test = X_tensor + noise_std * torch.randn_like(X_tensor)
with torch.no_grad():
    X_denoised = model(X_noisy_test)

noisy_mse   = ((X_noisy_test - X_tensor)**2).mean().item()
denoised_mse= ((X_denoised   - X_tensor)**2).mean().item()
print(f"\n  Noisy MSE:    {noisy_mse:.4f}")
print(f"  Denoised MSE: {denoised_mse:.4f}")
print(f"  SNR improvement: {10*np.log10(noisy_mse/denoised_mse):.1f} dB")

# ─── Visualize 4 examples ─────────────────────────────────────────────────────
t = np.linspace(0, 2*np.pi, 32)
fig, axes = plt.subplots(4, 1, figsize=(10, 8))
for i, ax in enumerate(axes):
    clean  = X_tensor[i].numpy()
    noisy  = X_noisy_test[i].numpy()
    denoised= X_denoised[i].numpy()
    ax.plot(t, clean,    "k-",  linewidth=2,   alpha=0.8, label="Clean")
    ax.plot(t, noisy,    "r--", linewidth=1,   alpha=0.6, label="Noisy")
    ax.plot(t, denoised, "b-",  linewidth=1.5, alpha=0.9, label="Denoised")
    ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.3)
    err_n = ((noisy - clean)**2).mean(); err_d = ((denoised - clean)**2).mean()
    ax.set_ylabel(f"Signal {i+1}\nMSE: noisy={err_n:.3f} denoised={err_d:.3f}", fontsize=7)
axes[-1].set_xlabel("Time")

plt.suptitle(f"Denoising AE (noise_std={noise_std})", fontsize=11)
plt.tight_layout(); plt.savefig("denoising_ae.png", dpi=85); plt.close()
print("Saved denoising_ae.png")
