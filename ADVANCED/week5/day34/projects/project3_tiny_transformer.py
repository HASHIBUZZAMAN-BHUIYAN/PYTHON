"""
Project: End-to-End TinyTransformer Trainer
Teaches: building, training, and evaluating a complete Transformer classifier
         with training curves, attention visualization, and parameter analysis.
~80 MB RAM, ~15s on CPU
"""
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

torch.manual_seed(42); np.random.seed(42)

VOCAB=20; D=16; H=4; FF=32; SEQ=8; N_CLASSES=3; LAYERS=2

# ─── Dataset: 600 samples, 3 classes ─────────────────────────────────────────
X=torch.cat([torch.randint(0,7,(200,SEQ)),torch.randint(7,14,(200,SEQ)),torch.randint(14,20,(200,SEQ))])
y=torch.cat([torch.zeros(200),torch.ones(200),torch.full((200,),2)]).long()
perm=torch.randperm(600); X,y=X[perm],y[perm]
X_tr,X_te,y_tr,y_te=train_test_split(X.numpy(),y.numpy(),test_size=0.2,random_state=42)
X_tr=torch.tensor(X_tr); X_te=torch.tensor(X_te)
y_tr=torch.tensor(y_tr); y_te=torch.tensor(y_te)

class SinPE(nn.Module):
    def __init__(self,d=D,M=50):
        super().__init__()
        pe=torch.zeros(M,d); pos=torch.arange(M).float().unsqueeze(1)
        div=torch.exp(torch.arange(0,d,2).float()*(-np.log(10000)/d))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe",pe.unsqueeze(0))
    def forward(self,x): return x+self.pe[:,:x.size(1)]

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn=nn.MultiheadAttention(D,H,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(D,FF),nn.GELU(),nn.Linear(FF,D))
        self.n1=nn.LayerNorm(D); self.n2=nn.LayerNorm(D)
    def forward(self,x):
        a,w=self.attn(x,x,x)
        x=self.n1(x+a); x=self.n2(x+self.ff(x))
        return x,w

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(VOCAB,D); self.pe=SinPE()
        self.blocks=nn.ModuleList([Block() for _ in range(LAYERS)])
        self.fc=nn.Linear(D,N_CLASSES)
    def forward(self,x,return_attn=False):
        h=self.pe(self.embed(x)); attns=[]
        for blk in self.blocks:
            h,w=blk(h); attns.append(w)
        out=self.fc(h.mean(1))
        return (out,attns) if return_attn else out

model=TinyTransformer()
n_params=sum(p.numel() for p in model.parameters())
print(f"=== TinyTransformer: {n_params:,} parameters ===\n")

opt=optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=200)
crit=nn.CrossEntropyLoss(); EPOCHS=200; BATCH=64

tr_losses,te_accs=[],[]
for epoch in range(EPOCHS):
    model.train()
    idx=torch.randperm(len(X_tr))[:BATCH]
    loss=crit(model(X_tr[idx]),y_tr[idx])
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    tr_losses.append(loss.item())
    if (epoch+1)%50==0:
        model.eval()
        with torch.no_grad(): acc=(model(X_te).argmax(-1)==y_te).float().mean().item()
        te_accs.append(acc)
        print(f"  Epoch {epoch+1}  loss={loss.item():.4f}  test_acc={acc:.3f}")

model.eval()
with torch.no_grad():
    logits,attns=model(X_te,return_attn=True)
    y_pred=logits.argmax(-1).numpy()
print("\n"+classification_report(y_te.numpy(),y_pred,target_names=["Low","Mid","High"],zero_division=0))

# ─── Visualize ────────────────────────────────────────────────────────────────
fig,axes=plt.subplots(2,2,figsize=(11,8))
axes[0,0].plot(tr_losses,alpha=0.8,color="steelblue"); axes[0,0].set_title("Training Loss"); axes[0,0].set_xlabel("Step")
axes[0,1].plot([50,100,150,200],te_accs,"o-",color="tomato"); axes[0,1].set_title("Test Accuracy"); axes[0,1].set_ylim(0,1.05)

# Attention from layer 0
w0=attns[0][0].detach().numpy()
im=axes[1,0].imshow(w0,cmap="viridis"); plt.colorbar(im,ax=axes[1,0])
axes[1,0].set_title("Layer 1 Attention (sample 0)")
# Attention from layer 1
w1=attns[1][0].detach().numpy()
im2=axes[1,1].imshow(w1,cmap="viridis"); plt.colorbar(im2,ax=axes[1,1])
axes[1,1].set_title("Layer 2 Attention (sample 0)")
plt.suptitle(f"TinyTransformer Results (acc={te_accs[-1]:.3f})",fontsize=11)
plt.tight_layout(); plt.savefig("tiny_transformer.png",dpi=85); plt.close()
print("Saved tiny_transformer.png")
