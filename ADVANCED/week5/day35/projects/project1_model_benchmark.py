"""
Project: Complete DL Architecture Benchmark
Teaches: systematic benchmarking of 5 architectures with a full dashboard.
~120 MB RAM, ~20s on CPU
"""
import time, numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

torch.manual_seed(42); np.random.seed(42)
VOCAB=20; D=16; SEQ=8; N=3; EPOCHS=200; BATCH=64

X=torch.cat([torch.randint(0,7,(200,SEQ)),torch.randint(7,14,(200,SEQ)),torch.randint(14,20,(200,SEQ))])
y=torch.cat([torch.zeros(200),torch.ones(200),torch.full((200,),2)]).long()
perm=torch.randperm(600); X,y=X[perm],y[perm]
X_tr,y_tr=X[:480],y[:480]; X_te,y_te=X[480:],y[480:]

class RNN(nn.Module):
    def __init__(self): super().__init__(); self.e=nn.Embedding(VOCAB,D); self.r=nn.RNN(D,32,batch_first=True); self.f=nn.Linear(32,N)
    def forward(self,x): _,h=self.r(self.e(x)); return self.f(h[-1])
class LSTM(nn.Module):
    def __init__(self): super().__init__(); self.e=nn.Embedding(VOCAB,D); self.r=nn.LSTM(D,32,batch_first=True); self.f=nn.Linear(32,N)
    def forward(self,x): _,(h,_)=self.r(self.e(x)); return self.f(h[-1])
class GRU(nn.Module):
    def __init__(self): super().__init__(); self.e=nn.Embedding(VOCAB,D); self.r=nn.GRU(D,32,batch_first=True); self.f=nn.Linear(32,N)
    def forward(self,x): _,h=self.r(self.e(x)); return self.f(h[-1])
class BiLSTM(nn.Module):
    def __init__(self): super().__init__(); self.e=nn.Embedding(VOCAB,D); self.r=nn.LSTM(D,16,batch_first=True,bidirectional=True); self.f=nn.Linear(32,N)
    def forward(self,x): _,(h,_)=self.r(self.e(x)); return self.f(torch.cat([h[0],h[1]],-1))
class Transformer(nn.Module):
    def __init__(self):
        super().__init__(); self.e=nn.Embedding(VOCAB,D)
        self.mha=nn.MultiheadAttention(D,4,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(D,32),nn.GELU(),nn.Linear(32,D))
        self.n1=nn.LayerNorm(D); self.n2=nn.LayerNorm(D); self.f=nn.Linear(D,N)
    def forward(self,x):
        e=self.e(x); a,_=self.mha(e,e,e); e=self.n1(e+a); e=self.n2(e+self.ff(e)); return self.f(e.mean(1))

MODELS={"RNN":RNN,"LSTM":LSTM,"GRU":GRU,"BiLSTM":BiLSTM,"Transformer":Transformer}
results={}

print(f"{'Model':<14} {'Acc':>6} {'Time':>7} {'Params':>8}")
print("─"*42)
for name,cls in MODELS.items():
    m=cls(); opt=optim.Adam(m.parameters(),lr=1e-3); crit=nn.CrossEntropyLoss()
    ls=[]; t0=time.time()
    for _ in range(EPOCHS):
        i=torch.randperm(480)[:BATCH]; loss=crit(m(X_tr[i]),y_tr[i])
        opt.zero_grad(); loss.backward(); opt.step(); ls.append(loss.item())
    elapsed=time.time()-t0
    m.eval()
    with torch.no_grad(): preds=m(X_te).argmax(-1)
    acc=(preds==y_te).float().mean().item()
    results[name]={"acc":acc,"time":elapsed,"losses":ls,"preds":preds.numpy(),"params":sum(p.numel() for p in m.parameters())}
    print(f"  {name:<12} {acc:>6.3f} {elapsed:>7.2f}s {results[name]['params']:>8,}")

# ─── Dashboard ────────────────────────────────────────────────────────────────
names=list(results.keys()); colors=[f"C{i}" for i in range(len(names))]
fig,axes=plt.subplots(2,3,figsize=(14,8))
for i,(n,r) in enumerate(results.items()): axes[0,0].plot(r["losses"],label=n,color=colors[i],alpha=0.8)
axes[0,0].set_title("Training Loss"); axes[0,0].legend(fontsize=8)
accs=[results[n]["acc"] for n in names]
axes[0,1].bar(names,accs,color=colors,alpha=0.85,edgecolor="k")
for i,v in enumerate(accs): axes[0,1].text(i,v+0.01,f"{v:.2f}",ha="center",fontsize=9)
axes[0,1].set_ylim(0,1.15); axes[0,1].set_title("Test Accuracy")
times=[results[n]["time"] for n in names]
axes[0,2].bar(names,times,color=colors,alpha=0.85,edgecolor="k")
axes[0,2].set_title("Training Time (s)")
params=[results[n]["params"] for n in names]
axes[1,0].bar(names,params,color=colors,alpha=0.85,edgecolor="k"); axes[1,0].set_title("Parameters")
# Acc per class for best model
best=max(results,key=lambda k:results[k]["acc"])
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_te.numpy(),results[best]["preds"])
axes[1,1].imshow(cm,cmap="Blues"); axes[1,1].set_title(f"Confusion: {best}")
for i in range(3):
    for j in range(3): axes[1,1].text(j,i,cm[i,j],ha="center",va="center",fontsize=10)
# Efficiency scatter
axes[1,2].scatter(params,accs,c=colors,s=100,edgecolors="k",zorder=3)
for i,n in enumerate(names): axes[1,2].annotate(n,(params[i],accs[i]),textcoords="offset points",xytext=(5,5),fontsize=8)
axes[1,2].set_xlabel("Parameters"); axes[1,2].set_ylabel("Accuracy"); axes[1,2].set_title("Efficiency")
plt.suptitle("DL Architecture Benchmark Dashboard",fontsize=12)
plt.tight_layout(); plt.savefig("benchmark.png",dpi=85); plt.close(); print("\nSaved benchmark.png")
