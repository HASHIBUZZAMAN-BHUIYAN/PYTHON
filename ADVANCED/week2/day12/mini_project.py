# Advanced Day 12 Mini-Project — Ablation Study: Training Techniques
# Systematically tests the effect of each tuning technique on digit classification.
# ~400 MB RAM, ~10 min on CPU

import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(42); np.random.seed(42)

digits = load_digits()
X = digits.data.astype(np.float32) / 16.0; y = digits.target
Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
Xtt  = torch.FloatTensor(Xtr).view(-1,1,8,8); ytt  = torch.LongTensor(ytr)
Xtet = torch.FloatTensor(Xte).view(-1,1,8,8); ytet = torch.LongTensor(yte)

class TunableCNN(nn.Module):
    def __init__(self, bn=True, drop=0.3, aug=False):
        super().__init__()
        self.aug = aug
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16) if bn else nn.Identity(), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32) if bn else nn.Identity(), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.clf = nn.Sequential(nn.Flatten(), nn.Linear(32*2*2,64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64,10))
    def forward(self, x):
        if self.aug and self.training:
            x = torch.clamp(x + torch.randn_like(x)*0.04, 0, 1)
        return self.clf(self.net(x))

configs = {
    "Baseline (no BN, no drop, no aug)": dict(bn=False, drop=0.0, aug=False),
    "+ BatchNorm":                        dict(bn=True,  drop=0.0, aug=False),
    "+ BatchNorm + Dropout":              dict(bn=True,  drop=0.3, aug=False),
    "+ BatchNorm + Dropout + Aug":        dict(bn=True,  drop=0.3, aug=True),
}

EPOCHS = 25
results_acc, results_gap = {}, {}

for name, cfg in configs.items():
    model = TunableCNN(**cfg)
    opt   = optim.Adam(model.parameters(), 1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    crit  = nn.CrossEntropyLoss()
    te_hist, tr_hist = [], []
    for ep in range(EPOCHS):
        model.train()
        for xb,yb in DataLoader(TensorDataset(Xtt,ytt), 32, shuffle=True):
            opt.zero_grad(); l=crit(model(xb),yb); l.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            tr_acc = (model(Xtt).argmax(1)==ytt).float().mean().item()
            te_acc = (model(Xtet).argmax(1)==ytet).float().mean().item()
        tr_hist.append(tr_acc); te_hist.append(te_acc)
    final_te = te_hist[-1]; final_gap = tr_hist[-1] - te_hist[-1]
    results_acc[name] = te_hist
    results_gap[name] = final_gap
    print(f"{name:<50}: test={final_te:.4f}  overfit_gap={final_gap:.4f}")

# ─── Report ───────────────────────────────────────────────────────────────────
print("\n=== Ablation Summary ===")
for name, te_hist in results_acc.items():
    print(f"  {name:<50}: final_acc={te_hist[-1]:.4f}  overfit_gap={results_gap[name]:.4f}")

# ─── Visualise ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Ablation Study: Effect of Training Techniques")

for name, te_hist in results_acc.items():
    axes[0].plot(te_hist, label=name.split("(")[0].strip())
axes[0].set_title("Test Accuracy over Epochs"); axes[0].set_xlabel("Epoch")
axes[0].legend(fontsize=7)

names = [n.split("(")[0].strip().replace("+ ","") for n in results_acc]
final_accs = [h[-1] for h in results_acc.values()]
gaps  = list(results_gap.values())
x = range(len(names))
axes[1].bar(x, final_accs, 0.4, label="Final test acc", color="steelblue")
axes[1].bar([i+0.4 for i in x], gaps, 0.4, label="Overfit gap", color="salmon")
axes[1].set_xticks([i+0.2 for i in x]); axes[1].set_xticklabels(names, rotation=20, fontsize=8)
axes[1].legend(); axes[1].set_title("Accuracy vs Overfitting")

plt.tight_layout(); plt.savefig("ablation_study.png", dpi=80)
print("\nSaved ablation_study.png")
plt.show()
