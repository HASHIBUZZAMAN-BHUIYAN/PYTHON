# Advanced Day 34 — Solutions
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42); np.random.seed(42)

# Data
X=torch.cat([torch.randint(0,7,(100,8)),torch.randint(7,14,(100,8)),torch.randint(14,20,(100,8))])
y=torch.cat([torch.zeros(100),torch.ones(100),torch.full((100,),2)]).long()
idx=torch.randperm(300); X,y=X[idx],y[idx]

class SinPE(nn.Module):
    def __init__(self,d,M=100):
        super().__init__()
        pe=torch.zeros(M,d); pos=torch.arange(M).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d,2).float()*(-np.log(10000)/d))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe",pe.unsqueeze(0))
    def forward(self,x): return x+self.pe[:,:x.size(1)]

class LearnedPE(nn.Module):
    def __init__(self,d,M=100): super().__init__(); self.pe=nn.Embedding(M,d)
    def forward(self,x): return x+self.pe(torch.arange(x.size(1),device=x.device))

class Block(nn.Module):
    def __init__(self,d,h,ff,residual=True,norm=True):
        super().__init__()
        self.attn=nn.MultiheadAttention(d,h,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(d,ff),nn.GELU(),nn.Linear(ff,d))
        self.n1=nn.LayerNorm(d) if norm else nn.Identity()
        self.n2=nn.LayerNorm(d) if norm else nn.Identity()
        self.res=residual
    def forward(self,x):
        a,_=self.attn(x,x,x)
        x=self.n1(x+a if self.res else a)
        f=self.ff(x)
        return self.n2(x+f if self.res else f)

class TT(nn.Module):
    def __init__(self,n_blocks=2,d=16,h=4,ff=32,pe_type="sin",residual=True,norm_on=True):
        super().__init__()
        self.embed=nn.Embedding(20,d)
        self.pe=(SinPE(d) if pe_type=="sin" else LearnedPE(d))
        self.blocks=nn.Sequential(*[Block(d,h,ff,residual,norm_on) for _ in range(n_blocks)])
        self.fc=nn.Linear(d,3)
    def forward(self,x): return self.fc(self.blocks(self.pe(self.embed(x))).mean(1))

def train_eval(model,epochs=150):
    opt=optim.Adam(model.parameters(),lr=1e-3); crit=nn.CrossEntropyLoss(); hist=[]
    for ep in range(epochs):
        i=torch.randperm(300)[:64]; loss=crit(model(X[i]),y[i])
        opt.zero_grad(); loss.backward(); opt.step(); hist.append(loss.item())
    with torch.no_grad(): acc=(model(X).argmax(-1)==y).float().mean().item()
    return acc, hist

# Ex 1
print("=== Ex 1: Learned vs Sinusoidal PE ===")
acc_sin,h_sin=train_eval(TT(pe_type="sin"))
acc_lrn,h_lrn=train_eval(TT(pe_type="learned"))
print(f"  Sinusoidal PE acc: {acc_sin:.3f}  Learned PE acc: {acc_lrn:.3f}")

# Ex 2
print("\n=== Ex 2: No Residuals ===")
acc_res,_=train_eval(TT(residual=True))
acc_nor,_=train_eval(TT(residual=False))
print(f"  With residuals: {acc_res:.3f}  Without: {acc_nor:.3f}")

# Ex 3
print("\n=== Ex 3: No LayerNorm ===")
acc_norm,_=train_eval(TT(norm_on=True))
acc_nonorm,_=train_eval(TT(norm_on=False))
print(f"  With LayerNorm: {acc_norm:.3f}  Without: {acc_nonorm:.3f}")

# Ex 4
print("\n=== Ex 4: Depth vs Width ===")
wide=TT(n_blocks=1,d=32,h=4,ff=64); deep=TT(n_blocks=4,d=8,h=2,ff=16)
acc_w,_=train_eval(wide); acc_d,_=train_eval(deep)
print(f"  Wide (1 block, d=32): params={sum(p.numel() for p in wide.parameters())}  acc={acc_w:.3f}")
print(f"  Deep (4 blocks, d=8): params={sum(p.numel() for p in deep.parameters())}  acc={acc_d:.3f}")

# Ex 5
print("\n=== Ex 5: Causal Next-Token Prediction ===")
class CausalTT(nn.Module):
    def __init__(self,d=16,h=4,ff=32):
        super().__init__()
        self.embed=nn.Embedding(20,d); self.pe=SinPE(d)
        self.block=Block(d,h,ff)
        self.fc=nn.Linear(d,20)
    def forward(self,x):
        S=x.size(1); mask=torch.triu(torch.ones(S,S),diagonal=1).bool()
        e=self.pe(self.embed(x)); a,_=self.block.attn(e,e,e,attn_mask=mask.float()*-1e9)
        e=self.block.n1(e+a); f=self.block.ff(e); e=self.block.n2(e+f)
        return self.fc(e)

causal=CausalTT(); opt=optim.Adam(causal.parameters(),lr=1e-3); crit=nn.CrossEntropyLoss()
for _ in range(200):
    seq=torch.randint(0,20,(32,8)); logits=causal(seq)
    loss=crit(logits[:,:-1].reshape(-1,20),seq[:,1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
causal.eval()
print("  Sample predictions (last token):")
for _ in range(5):
    seq=torch.randint(0,20,(1,8))
    with torch.no_grad(): nxt=causal(seq)[0,-1].argmax().item()
    print(f"    Input: {seq[0].tolist()} → next: {nxt}")
