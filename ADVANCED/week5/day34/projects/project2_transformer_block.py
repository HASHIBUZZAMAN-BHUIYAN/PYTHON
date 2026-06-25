"""
Project: Transformer Block Component Ablation
Teaches: ablating each component of a Transformer block individually
         to see which matters most: attention, FFN, residuals, LayerNorm.
~60 MB RAM, ~10s on CPU
"""
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42); np.random.seed(42)

X=torch.cat([torch.randint(0,7,(120,8)),torch.randint(7,14,(120,8)),torch.randint(14,20,(120,8))])
y=torch.cat([torch.zeros(120),torch.ones(120),torch.full((120,),2)]).long()
perm=torch.randperm(360); X,y=X[perm],y[perm]

D,H,FF=16,4,32

class TransformerBlock(nn.Module):
    def __init__(self, use_attn=True, use_ffn=True, use_residual=True, use_norm=True):
        super().__init__()
        self.attn=nn.MultiheadAttention(D,H,batch_first=True) if use_attn else None
        self.ff  =nn.Sequential(nn.Linear(D,FF),nn.GELU(),nn.Linear(FF,D)) if use_ffn else None
        self.n1  =nn.LayerNorm(D) if use_norm else nn.Identity()
        self.n2  =nn.LayerNorm(D) if use_norm else nn.Identity()
        self.residual=use_residual
    def forward(self,x):
        if self.attn:
            a,_=self.attn(x,x,x)
            x=self.n1(x+a if self.residual else a)
        if self.ff:
            f=self.ff(x)
            x=self.n2(x+f if self.residual else f)
        return x

class Model(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embed=nn.Embedding(20,D)
        self.block=TransformerBlock(**kwargs)
        self.fc=nn.Linear(D,3)
    def forward(self,x): return self.fc(self.block(self.embed(x)).mean(1))

CONFIGS = {
    "Full":          dict(use_attn=True,  use_ffn=True,  use_residual=True,  use_norm=True),
    "No FFN":        dict(use_attn=True,  use_ffn=False, use_residual=True,  use_norm=True),
    "No Attn":       dict(use_attn=False, use_ffn=True,  use_residual=True,  use_norm=True),
    "No Residual":   dict(use_attn=True,  use_ffn=True,  use_residual=False, use_norm=True),
    "No LayerNorm":  dict(use_attn=True,  use_ffn=True,  use_residual=True,  use_norm=False),
}

def train_eval(config, epochs=150):
    model=Model(**config); opt=optim.Adam(model.parameters(),lr=1e-3); crit=nn.CrossEntropyLoss()
    hist=[]
    for _ in range(epochs):
        i=torch.randperm(360)[:64]; loss=crit(model(X[i]),y[i])
        opt.zero_grad(); loss.backward(); opt.step(); hist.append(loss.item())
    with torch.no_grad(): acc=(model(X).argmax(-1)==y).float().mean().item()
    return acc, hist

print("=== Transformer Block Ablation Study ===\n")
results = {}
for name, cfg in CONFIGS.items():
    acc, hist = train_eval(cfg)
    results[name] = {"acc": acc, "hist": hist}
    print(f"  {name:<20}: acc={acc:.3f}")

# ─── Visualize ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# Accuracy bar chart
names = list(results.keys()); accs = [results[n]["acc"] for n in names]
colors=["green" if n=="Full" else "tomato" for n in names]
axes[0].bar(names, accs, color=colors, alpha=0.85, edgecolor="k", linewidth=0.7)
axes[0].set_ylim(0,1.05); axes[0].set_title("Test Accuracy by Configuration")
axes[0].set_xticklabels(names, rotation=30, ha="right")
for i,v in enumerate(accs): axes[0].text(i,v+0.01,f"{v:.2f}",ha="center",fontsize=9)
# Loss curves
for name, r in results.items():
    lw = 2.5 if name=="Full" else 1
    axes[1].plot(r["hist"], label=name, alpha=0.8, linewidth=lw)
axes[1].set_xlabel("Step"); axes[1].set_ylabel("Loss"); axes[1].set_title("Training Loss Curves")
axes[1].legend(fontsize=8); axes[1].set_ylim(0,2.5)
plt.suptitle("Transformer Block Ablation", fontsize=11)
plt.tight_layout(); plt.savefig("block_ablation.png",dpi=85); plt.close()
print("\nSaved block_ablation.png")
# Summary
best=max(results,key=lambda k:results[k]["acc"])
worst=min(results,key=lambda k:results[k]["acc"])
print(f"Best config:  {best}  ({results[best]['acc']:.3f})")
print(f"Worst config: {worst} ({results[worst]['acc']:.3f})")
