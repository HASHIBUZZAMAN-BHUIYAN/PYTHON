# Advanced Day 35 — Solutions
import time, numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42); np.random.seed(42)
VOCAB=20; D=16; SEQ=8; N_CLASSES=3; BATCH=64; EPOCHS=100

def make_data(seq_len=8, n=600):
    X=torch.cat([torch.randint(0,7,(n//3,seq_len)),torch.randint(7,14,(n//3,seq_len)),torch.randint(14,20,(n//3,seq_len))])
    y=torch.cat([torch.zeros(n//3),torch.ones(n//3),torch.full((n//3,),2)]).long()
    perm=torch.randperm(n); return X[perm],y[perm]

X,y=make_data(); n_tr=480; X_tr,y_tr=X[:n_tr],y[:n_tr]; X_te,y_te=X[n_tr:],y[n_tr:]

class RNN(nn.Module):
    def __init__(self): super().__init__(); self.e=nn.Embedding(VOCAB,D); self.r=nn.RNN(D,32,batch_first=True); self.f=nn.Linear(32,N_CLASSES)
    def forward(self,x): _,h=self.r(self.e(x)); return self.f(h.squeeze(0))
class LSTM(nn.Module):
    def __init__(self): super().__init__(); self.e=nn.Embedding(VOCAB,D); self.r=nn.LSTM(D,32,batch_first=True); self.f=nn.Linear(32,N_CLASSES)
    def forward(self,x): _,(h,_)=self.r(self.e(x)); return self.f(h.squeeze(0))
class BiLSTM(nn.Module):
    def __init__(self): super().__init__(); self.e=nn.Embedding(VOCAB,D); self.r=nn.LSTM(D,16,batch_first=True,bidirectional=True); self.f=nn.Linear(32,N_CLASSES)
    def forward(self,x): _,(h,_)=self.r(self.e(x)); return self.f(torch.cat([h[0],h[1]],dim=-1))

def train(m,X_tr,y_tr,X_te,y_te,lr=1e-3,epochs=EPOCHS):
    opt=optim.Adam(m.parameters(),lr=lr); crit=nn.CrossEntropyLoss(); ls=[]
    for _ in range(epochs):
        i=torch.randperm(len(X_tr))[:BATCH]; loss=crit(m(X_tr[i]),y_tr[i])
        opt.zero_grad(); loss.backward(); opt.step(); ls.append(loss.item())
    m.eval()
    with torch.no_grad(): acc=(m(X_te).argmax(-1)==y_te).float().mean().item()
    return acc,ls

# Ex 1
print("=== Ex 1: BiLSTM ===")
acc_l,_=train(LSTM(),X_tr,y_tr,X_te,y_te)
acc_b,_=train(BiLSTM(),X_tr,y_tr,X_te,y_te)
print(f"  LSTM: {acc_l:.3f}  BiLSTM: {acc_b:.3f}")

# Ex 2
print("\n=== Ex 2: LR Sensitivity ===")
fig,axes=plt.subplots(3,3,figsize=(12,9))
for mi,(mname,mfn) in enumerate([("RNN",RNN),("LSTM",LSTM),("BiLSTM",BiLSTM)]):
    for li,lr in enumerate([1e-4,1e-3,1e-2]):
        m=mfn(); acc,ls=train(m,X_tr,y_tr,X_te,y_te,lr=lr,epochs=80)
        axes[mi,li].plot(ls,alpha=0.8); axes[mi,li].set_title(f"{mname} lr={lr:.0e} acc={acc:.2f}",fontsize=8)
plt.tight_layout(); plt.savefig("lr_sensitivity.png",dpi=80); plt.close(); print("  Saved lr_sensitivity.png")

# Ex 3
print("\n=== Ex 3: Sequence Length Sensitivity ===")
for seq_len in [4,8,16]:
    Xl,yl=make_data(seq_len=seq_len)
    Xtr,ytr=Xl[:480],yl[:480]; Xte,yte=Xl[480:],yl[480:]
    for mname,mfn in [("RNN",RNN),("LSTM",LSTM)]:
        m=mfn(); acc,_=train(m,Xtr,ytr,Xte,yte,epochs=100)
        print(f"  seq={seq_len}  {mname}: {acc:.3f}")

# Ex 4
print("\n=== Ex 4: Noise Robustness ===")
def add_noise(X,noise_frac=0.2):
    Xn=X.clone()
    mask=torch.rand_like(X.float())<noise_frac
    Xn[mask]=torch.randint(0,VOCAB,(mask.sum().item(),))
    return Xn

X_noisy=add_noise(X_te,0.2)
for mname,mfn in [("RNN",RNN),("LSTM",LSTM),("BiLSTM",BiLSTM)]:
    m=mfn(); train(m,X_tr,y_tr,X_te,y_te,epochs=120)
    m.eval()
    with torch.no_grad():
        acc_clean=(m(X_te).argmax(-1)==y_te).float().mean().item()
        acc_noisy=(m(X_noisy).argmax(-1)==y_te).float().mean().item()
    print(f"  {mname}: clean={acc_clean:.3f}  noisy={acc_noisy:.3f}  drop={acc_clean-acc_noisy:.3f}")

# Ex 5
print("\n=== Ex 5: FLOPs Estimate ===")
def rnn_flops(seq,d_in,d_hidden): return seq*(d_in*d_hidden + d_hidden*d_hidden)*2
def lstm_flops(seq,d_in,d_hidden): return 4*rnn_flops(seq,d_in,d_hidden)
macs={
    "RNN" :  D*VOCAB + rnn_flops(SEQ,D,32) + 32*N_CLASSES,
    "LSTM":  D*VOCAB + lstm_flops(SEQ,D,32) + 32*N_CLASSES,
    "BiLSTM":D*VOCAB + 2*lstm_flops(SEQ,D,16) + 32*N_CLASSES,
}
for name,mc in sorted(macs.items(),key=lambda x:x[1]): print(f"  {name:<10}: {mc:>8,} MACs")
