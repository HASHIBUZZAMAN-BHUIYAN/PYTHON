"""
Project: Architecture Tradeoff Analysis
Teaches: plotting Pareto frontiers (accuracy vs params, accuracy vs speed),
         discussing when to use each architecture.
~80 MB RAM, ~15s on CPU
"""
import time, numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42); np.random.seed(42)
VOCAB=20; D=16; SEQ=8; N=3; EPOCHS=150; BATCH=64

X=torch.cat([torch.randint(0,7,(200,SEQ)),torch.randint(7,14,(200,SEQ)),torch.randint(14,20,(200,SEQ))])
y=torch.cat([torch.zeros(200),torch.ones(200),torch.full((200,),2)]).long()
perm=torch.randperm(600); X,y=X[perm],y[perm]
X_tr,y_tr=X[:480],y[:480]; X_te,y_te=X[480:],y[480:]

# Many architecture variants
VARIANTS = {
    "RNN-tiny":    lambda: nn.Sequential(nn.Embedding(VOCAB,D), *[]),
}

class Base(nn.Module):
    def __init__(self,rnn_cls,hidden,layers,bi=False):
        super().__init__()
        self.embed=nn.Embedding(VOCAB,D)
        self.rnn=rnn_cls(D,hidden,num_layers=layers,batch_first=True,
                         **({'bidirectional':bi} if bi else {}))
        out=hidden*2 if bi else hidden
        self.fc=nn.Linear(out,N)
        self.bi=bi; self.is_lstm=rnn_cls==nn.LSTM
    def forward(self,x):
        e=self.embed(x)
        if self.is_lstm: _,(h,_)=self.rnn(e)
        else: _,h=self.rnn(e)
        if self.bi: h=torch.cat([h[0],h[1]],-1)
        else: h=h[-1]
        return self.fc(h)

class TF(nn.Module):
    def __init__(self,d,h,ff,layers):
        super().__init__()
        self.embed=nn.Embedding(VOCAB,d)
        self.blocks=nn.ModuleList([
            nn.TransformerEncoderLayer(d,h,ff,dropout=0.0,batch_first=True,activation="gelu")
            for _ in range(layers)])
        self.fc=nn.Linear(d,N)
    def forward(self,x):
        e=self.embed(x)
        for b in self.blocks: e=b(e)
        return self.fc(e.mean(1))

MODELS = {
    "RNN-tiny":    Base(nn.RNN,  8, 1),
    "RNN-base":    Base(nn.RNN, 32, 1),
    "LSTM-tiny":   Base(nn.LSTM, 8, 1),
    "LSTM-base":   Base(nn.LSTM,32, 1),
    "LSTM-deep":   Base(nn.LSTM,16, 3),
    "GRU-base":    Base(nn.GRU, 32, 1),
    "BiLSTM-base": Base(nn.LSTM,16, 1,bi=True),
    "TF-tiny":     TF(8,  2, 16, 1),
    "TF-base":     TF(16, 4, 32, 1),
    "TF-deep":     TF(16, 4, 32, 3),
}

results={}
print("Benchmarking all variants ...")
for name, model in MODELS.items():
    opt=optim.Adam(model.parameters(),lr=1e-3); crit=nn.CrossEntropyLoss()
    t0=time.time()
    for _ in range(EPOCHS):
        i=torch.randperm(480)[:BATCH]; loss=crit(model(X_tr[i]),y_tr[i])
        opt.zero_grad(); loss.backward(); opt.step()
    elapsed=time.time()-t0
    model.eval()
    with torch.no_grad(): acc=(model(X_te).argmax(-1)==y_te).float().mean().item()
    n_p=sum(p.numel() for p in model.parameters())
    results[name]={"acc":acc,"time":elapsed,"params":n_p}
    print(f"  {name:<16}: acc={acc:.3f}  t={elapsed:.2f}s  params={n_p:,}")

# ─── Pareto analysis ─────────────────────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(13,5))
cats={"RNN":"steelblue","LSTM":"tomato","GRU":"limegreen","BiLSTM":"orange","TF":"purple"}

for name,r in results.items():
    cat=name.split("-")[0]
    c=cats.get(cat,"gray")
    axes[0].scatter(r["params"],r["acc"],color=c,s=80,zorder=3,edgecolors="k",linewidth=0.5)
    axes[0].annotate(name,(r["params"],r["acc"]),textcoords="offset points",xytext=(4,4),fontsize=7)
    axes[1].scatter(r["time"],r["acc"],color=c,s=80,zorder=3,edgecolors="k",linewidth=0.5)
    axes[1].annotate(name,(r["time"],r["acc"]),textcoords="offset points",xytext=(4,4),fontsize=7)

for cat,c in cats.items(): axes[0].scatter([],[],color=c,label=cat,s=60)
axes[0].set_xlabel("Parameters"); axes[0].set_ylabel("Test Accuracy"); axes[0].set_title("Accuracy vs Parameters")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
axes[1].set_xlabel("Training Time (s)"); axes[1].set_ylabel("Test Accuracy"); axes[1].set_title("Accuracy vs Speed")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

plt.suptitle("Architecture Tradeoff: Pareto Analysis",fontsize=11)
plt.tight_layout(); plt.savefig("pareto.png",dpi=85); plt.close(); print("\nSaved pareto.png")

# Summary
best_acc    =max(results,key=lambda k:results[k]["acc"])
best_speed  =min(results,key=lambda k:results[k]["time"])
best_eff    =max(results,key=lambda k:results[k]["acc"]/results[k]["params"])
print(f"\n  Best accuracy:    {best_acc}  ({results[best_acc]['acc']:.3f})")
print(f"  Fastest:          {best_speed}  ({results[best_speed]['time']:.2f}s)")
print(f"  Best efficiency:  {best_eff}  (acc/params = {results[best_eff]['acc']/results[best_eff]['params']:.6f})")
