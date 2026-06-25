# Advanced Day 35 — Week 5 Capstone: Model Comparison
# ~120 MB RAM, ~20s on CPU

print("""
=== Week 5 Capstone: Deep Learning Model Comparison — Day 35 ===

We'll compare all architectures from this week on a single benchmark:
  - RNN (vanilla)
  - LSTM
  - GRU
  - Autoencoder (for reconstruction)
  - TinyTransformer

Task: 3-class sequence classification (token ranges low/mid/high)
Metrics: accuracy, training speed, parameter count, memory
""")

import time, numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42); np.random.seed(42)

VOCAB=20; D=16; SEQ=8; N_CLASSES=3; BATCH=64; EPOCHS=150

# ─── Dataset ─────────────────────────────────────────────────────────────────
X=torch.cat([torch.randint(0,7,(200,SEQ)),torch.randint(7,14,(200,SEQ)),torch.randint(14,20,(200,SEQ))])
y=torch.cat([torch.zeros(200),torch.ones(200),torch.full((200,),2)]).long()
perm=torch.randperm(600); X,y=X[perm],y[perm]
n_train=480; X_tr,y_tr=X[:n_train],y[:n_train]; X_te,y_te=X[n_train:],y[n_train:]

# ─── Models ──────────────────────────────────────────────────────────────────
class RNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(VOCAB,D)
        self.rnn  =nn.RNN(D,32,batch_first=True)
        self.fc   =nn.Linear(32,N_CLASSES)
    def forward(self,x):
        _,h=self.rnn(self.embed(x)); return self.fc(h.squeeze(0))

class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(VOCAB,D)
        self.lstm =nn.LSTM(D,32,batch_first=True)
        self.fc   =nn.Linear(32,N_CLASSES)
    def forward(self,x):
        _,(h,_)=self.lstm(self.embed(x)); return self.fc(h.squeeze(0))

class GRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(VOCAB,D)
        self.gru  =nn.GRU(D,32,batch_first=True)
        self.fc   =nn.Linear(32,N_CLASSES)
    def forward(self,x):
        _,h=self.gru(self.embed(x)); return self.fc(h.squeeze(0))

class TransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(VOCAB,D)
        self.mha  =nn.MultiheadAttention(D,4,batch_first=True)
        self.ff   =nn.Sequential(nn.Linear(D,32),nn.GELU(),nn.Linear(32,D))
        self.norm1=nn.LayerNorm(D); self.norm2=nn.LayerNorm(D)
        self.fc   =nn.Linear(D,N_CLASSES)
    def forward(self,x):
        e=self.embed(x); a,_=self.mha(e,e,e); e=self.norm1(e+a)
        e=self.norm2(e+self.ff(e)); return self.fc(e.mean(1))

MODELS = {
    "RNN":         RNNClassifier(),
    "LSTM":        LSTMClassifier(),
    "GRU":         GRUClassifier(),
    "Transformer": TransformerClassifier(),
}

# ─── Training framework ───────────────────────────────────────────────────────
def train_model(name, model):
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit= nn.CrossEntropyLoss()
    losses=[]; t0=time.time()
    for epoch in range(EPOCHS):
        model.train()
        idx=torch.randperm(n_train)[:BATCH]
        loss=crit(model(X_tr[idx]),y_tr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    elapsed=time.time()-t0
    model.eval()
    with torch.no_grad():
        acc=(model(X_te).argmax(-1)==y_te).float().mean().item()
    n_params=sum(p.numel() for p in model.parameters())
    return {"losses":losses,"acc":acc,"time":elapsed,"params":n_params}

print("Training all models ...\n")
print(f"{'Model':<15}  {'Accuracy':>8}  {'Time(s)':>8}  {'Params':>8}")
print("-"*45)
results={}
for name, model in MODELS.items():
    r=train_model(name,model)
    results[name]=r
    print(f"  {name:<13}  {r['acc']:>8.3f}  {r['time']:>8.2f}  {r['params']:>8,}")

# ─── Visualize ────────────────────────────────────────────────────────────────
fig,axes=plt.subplots(2,2,figsize=(12,8))
colors=["steelblue","tomato","limegreen","purple"]
names=list(results.keys())

# Loss curves
for i,(name,r) in enumerate(results.items()):
    axes[0,0].plot(r["losses"],label=name,alpha=0.8,color=colors[i])
axes[0,0].set_title("Training Loss"); axes[0,0].set_xlabel("Step"); axes[0,0].legend()

# Accuracy bar
accs=[results[n]["acc"] for n in names]
axes[0,1].bar(names,accs,color=colors,alpha=0.85,edgecolor="k")
for i,v in enumerate(accs): axes[0,1].text(i,v+0.01,f"{v:.2f}",ha="center")
axes[0,1].set_title("Test Accuracy"); axes[0,1].set_ylim(0,1.1)

# Speed bar
times=[results[n]["time"] for n in names]
axes[1,0].bar(names,times,color=colors,alpha=0.85,edgecolor="k")
for i,v in enumerate(times): axes[1,0].text(i,v+0.01,f"{v:.2f}s",ha="center",fontsize=8)
axes[1,0].set_title("Training Time (seconds)")

# Params bar
params=[results[n]["params"] for n in names]
axes[1,1].bar(names,params,color=colors,alpha=0.85,edgecolor="k")
for i,v in enumerate(params): axes[1,1].text(i,v+20,str(v),ha="center",fontsize=8)
axes[1,1].set_title("Parameter Count")

plt.suptitle("Week 5 Capstone: DL Architecture Comparison",fontsize=12)
plt.tight_layout(); plt.savefig("day35_comparison.png",dpi=85); plt.close()
print("\nSaved day35_comparison.png")

best=max(results,key=lambda k:results[k]["acc"])
fastest=min(results,key=lambda k:results[k]["time"])
print(f"\n  Most accurate  : {best} ({results[best]['acc']:.3f})")
print(f"  Fastest training: {fastest} ({results[fastest]['time']:.2f}s)")
print(f"  Best efficiency: max acc/params = {max(results,key=lambda k:results[k]['acc']/results[k]['params'])}")
