"""
Project: Network Intrusion Anomaly Detector
Teaches: training an AE on "normal" network traffic features, then using
         reconstruction error to flag anomalous (intrusion) traffic.
~50 MB RAM, ~5s on CPU
"""
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve

torch.manual_seed(42); np.random.seed(42)

# ─── Synthetic network traffic features (10 dimensions) ──────────────────────
# Normal: low packet rate, normal port range, moderate payload
def make_normal(n=1000):
    X = np.random.randn(n, 10) * 0.5
    X[:,0] += 5.0   # avg packet rate  ~5 Mbps
    X[:,1]  = np.abs(X[:,1])*0.3 + 0.1  # packet loss < 5%
    X[:,2]  = np.random.uniform(0, 1, n)  # protocol variety
    X[:,3:6]+= np.random.uniform(0.3, 0.7, (n,3))  # port entropy
    return X.astype(np.float32)

# Anomaly: high packet rate burst (DDoS-like) or unusual port pattern
def make_anomaly(n=100):
    X_ddos = np.random.randn(n//2, 10) * 0.3
    X_ddos[:,0] += 50.0  # very high packet rate
    X_ddos[:,1]  = 0.8 + 0.1*np.random.rand(n//2)  # high packet loss
    X_scan = np.random.randn(n//2, 10) * 0.5
    X_scan[:,2]  = 0.0   # only one protocol
    X_scan[:,3:6]= 0.0   # all to one port
    return np.vstack([X_ddos, X_scan]).astype(np.float32)

X_normal  = make_normal(1000)
X_anomaly = make_anomaly(200)

# Normalize
mean, std = X_normal.mean(0), X_normal.std(0) + 1e-8
X_normal_n  = (X_normal  - mean) / std
X_anomaly_n = (X_anomaly - mean) / std

X_train = torch.tensor(X_normal_n[:800])
X_val   = torch.tensor(X_normal_n[800:])
X_anom  = torch.tensor(X_anomaly_n)

class NetworkAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(10,8), nn.ReLU(), nn.Linear(8,3))
        self.decoder = nn.Sequential(nn.Linear(3,8),  nn.ReLU(), nn.Linear(8,10))
    def forward(self, x):
        return self.decoder(self.encoder(x))

ae  = NetworkAE()
opt = optim.Adam(ae.parameters(), lr=1e-3)
crit= nn.MSELoss()

print("Training on normal traffic only ...")
for epoch in range(500):
    ae.train()
    loss = crit(ae(X_train), X_train)
    opt.zero_grad(); loss.backward(); opt.step()
    if (epoch+1) % 100 == 0:
        ae.eval()
        with torch.no_grad(): val_loss = crit(ae(X_val), X_val)
        print(f"  Epoch {epoch+1}  train={loss.item():.4f}  val={val_loss.item():.4f}")

ae.eval()
with torch.no_grad():
    norm_recon = ((ae(X_val) - X_val)**2).mean(dim=1).numpy()
    anom_recon = ((ae(X_anom)- X_anom)**2).mean(dim=1).numpy()

threshold = np.percentile(norm_recon, 95)
detected  = np.sum(anom_recon > threshold)
fp        = np.sum(norm_recon > threshold)

print(f"\n  Normal mean error  : {norm_recon.mean():.4f}")
print(f"  Anomaly mean error : {anom_recon.mean():.4f}")
print(f"  Threshold (95th%ile): {threshold:.4f}")
print(f"  Anomalies detected : {detected}/{len(anom_recon)} = {detected/len(anom_recon):.1%}")
print(f"  False positives    : {fp}/{len(norm_recon)} = {fp/len(norm_recon):.1%}")

# ─── ROC AUC ─────────────────────────────────────────────────────────────────
all_err    = np.concatenate([norm_recon, anom_recon])
all_labels = np.array([0]*len(norm_recon) + [1]*len(anom_recon))
auc = roc_auc_score(all_labels, all_err)
print(f"  ROC-AUC: {auc:.3f}")

# ─── Visualize ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(norm_recon, bins=30, alpha=0.7, label="Normal", color="steelblue")
axes[0].hist(anom_recon, bins=30, alpha=0.7, label="Anomaly", color="tomato")
axes[0].axvline(threshold, color="k", linestyle="--", linewidth=2, label=f"Thresh={threshold:.3f}")
axes[0].set_xlabel("Reconstruction Error"); axes[0].set_title("Error Distribution")
axes[0].legend()

axes[1].scatter(range(len(norm_recon)), norm_recon, c="steelblue", s=8, alpha=0.5, label="Normal")
axes[1].scatter(range(len(norm_recon), len(norm_recon)+len(anom_recon)),
                anom_recon, c="tomato", s=8, alpha=0.7, label="Anomaly")
axes[1].axhline(threshold, color="k", linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Sample"); axes[1].set_ylabel("Error"); axes[1].set_title("Per-Sample Errors")
axes[1].legend()

plt.suptitle(f"Network Anomaly Detector (AUC={auc:.3f})", fontsize=11)
plt.tight_layout(); plt.savefig("anomaly_detection.png", dpi=85); plt.close()
print("Saved anomaly_detection.png")
