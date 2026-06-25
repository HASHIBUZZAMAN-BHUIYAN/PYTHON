# DL Reference — Autoencoder Template
# Covers: reconstruction, denoising, anomaly detection, compression.
# ~40 MB RAM, <5s on CPU
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ─── 1. AUTOENCODER ARCHITECTURE ─────────────────────────────────────────────
class Autoencoder(nn.Module):
    """
    Symmetric autoencoder.
    encoder_dims: list of hidden dims from input to bottleneck
    Example: input=64, encoder_dims=[32,16,8] → bottleneck=8
    """
    def __init__(self, input_dim: int, encoder_dims: list, activation=nn.ReLU):
        super().__init__()
        dims      = [input_dim] + encoder_dims
        enc_layers= []
        for i in range(len(dims)-1):
            enc_layers += [nn.Linear(dims[i], dims[i+1]), activation()]
        self.encoder = nn.Sequential(*enc_layers[:-1])  # no activation after last

        dec_dims   = list(reversed(dims))
        dec_layers = []
        for i in range(len(dec_dims)-1):
            dec_layers += [nn.Linear(dec_dims[i], dec_dims[i+1]),
                           activation() if i < len(dec_dims)-2 else nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):   return self.encoder(x)
    def decode(self, z):   return self.decoder(z)
    def forward(self, x):  return self.decode(self.encode(x))

# ─── 2. TRAINING LOOP ─────────────────────────────────────────────────────────
def train_autoencoder(model, X_train: torch.Tensor, X_val: torch.Tensor,
                      n_epochs=30, batch_size=32, lr=1e-3, noise_std=0.):
    """
    noise_std > 0 → denoising autoencoder (adds Gaussian noise to inputs).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    ds  = torch.utils.data.TensorDataset(X_train, X_train)
    dl  = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    history = {"train": [], "val": []}
    for epoch in range(1, n_epochs+1):
        model.train(); total = 0.
        for xb, target in dl:
            inp = xb + noise_std * torch.randn_like(xb) if noise_std > 0 else xb
            loss = criterion(model(inp), target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * len(xb)
        tl = total / len(dl.dataset)
        with torch.no_grad():
            vl = criterion(model(X_val), X_val).item()
        history["train"].append(tl); history["val"].append(vl)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:>3}: train={tl:.5f}  val={vl:.5f}")
    return history

# ─── 3. ANOMALY DETECTION ────────────────────────────────────────────────────
def reconstruction_error(model, X: torch.Tensor) -> np.ndarray:
    """Per-sample MSE reconstruction error."""
    model.eval()
    with torch.no_grad():
        recon = model(X)
        errors = ((X - recon)**2).mean(dim=1).numpy()
    return errors

def anomaly_threshold(errors_normal: np.ndarray, percentile=95) -> float:
    return float(np.percentile(errors_normal, percentile))

# ─── 4. VISUALIZATION ─────────────────────────────────────────────────────────
def plot_reconstructions(model, X: torch.Tensor, n=8, img_shape=(8,8),
                          title="Reconstructions", save_path=None):
    model.eval()
    with torch.no_grad(): recon = model(X[:n])
    fig, axes = plt.subplots(2, n, figsize=(n*1.5, 3))
    for i in range(n):
        axes[0,i].imshow(X[i].numpy().reshape(img_shape), cmap="gray")
        axes[1,i].imshow(recon[i].numpy().reshape(img_shape), cmap="gray")
        for ax in axes[:,i]: ax.axis("off")
    axes[0,0].set_ylabel("Original", fontsize=8)
    axes[1,0].set_ylabel("Recon",    fontsize=8)
    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=80); plt.close()
    else: plt.show()

# ─── DEMO ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets import load_digits
    digits = load_digits()
    X = torch.from_numpy((digits.data / 16.).astype(np.float32))
    y = digits.target
    # Use digits 0&1 as "normal", digit 7 as "anomaly"
    normal_mask = (y==0)|(y==1)
    X_norm = X[normal_mask]; X_anom = X[y==7]
    split = int(len(X_norm)*0.8)
    model = Autoencoder(64, [32,16,8])
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    train_autoencoder(model, X_norm[:split], X_norm[split:], n_epochs=30)
    err_norm = reconstruction_error(model, X_norm[split:])
    err_anom = reconstruction_error(model, X_anom[:50])
    thresh   = anomaly_threshold(err_norm)
    print(f"Threshold: {thresh:.4f}")
    print(f"Normal  detected as anomaly: {(err_norm>thresh).mean()*100:.1f}%")
    print(f"Anomaly detected as anomaly: {(err_anom>thresh).mean()*100:.1f}%")
    plot_reconstructions(model, X_norm[:8], save_path="autoencoder_demo.png")
    print("Saved autoencoder_demo.png")
