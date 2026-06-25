# Advanced Day 33 — Solutions
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42); torch.manual_seed(42)

def softmax(x,axis=-1):
    e=np.exp(x-x.max(axis=axis,keepdims=True)); return e/e.sum(axis=axis,keepdims=True)

def sdpa(Q,K,V,mask=None,temp=1.0):
    d_k=Q.shape[-1]; scores=Q@K.T/(np.sqrt(d_k)*temp)
    if mask is not None: scores=np.where(mask,-1e9,scores)
    w=softmax(scores,-1); return w@V, w

# Ex 1
print("=== Ex 1: Temperature Scaling ===")
T=5; d=8; Q=np.random.randn(T,d); K=np.random.randn(T,d); V=np.random.randn(T,d)
fig,axes=plt.subplots(1,3,figsize=(12,3))
for ax,temp,label in zip(axes,[0.1,1.0,5.0],["spiky (0.1)","normal (1.0)","uniform (5.0)"]):
    _,w=sdpa(Q,K,V,temp=temp)
    ax.imshow(w,cmap="Reds",vmin=0,vmax=1); ax.set_title(label)
    for i in range(T):
        for j in range(T): ax.text(j,i,f"{w[i,j]:.2f}",ha="center",va="center",fontsize=7)
plt.tight_layout(); plt.savefig("temp_attn.png",dpi=80); plt.close(); print("  Saved temp_attn.png")

# Ex 2
print("\n=== Ex 2: Cross-Attention ===")
enc_seq,dec_seq,d=4,3,8
K_enc=np.random.randn(enc_seq,d); V_enc=np.random.randn(enc_seq,d); Q_dec=np.random.randn(dec_seq,d)
out,w=sdpa(Q_dec,K_enc,V_enc)
print(f"  Cross-attn output: {out.shape}  weights: {w.shape}")
print(f"  Weights (decoder × encoder):\n{w.round(3)}")

# Ex 3
print("\n=== Ex 3: MultiHead from Scratch ===")
class MHA(nn.Module):
    def __init__(self,d,h):
        super().__init__(); self.h=h; self.dk=d//h
        self.Wq=[nn.Linear(d,self.dk,bias=False) for _ in range(h)]
        self.Wk=[nn.Linear(d,self.dk,bias=False) for _ in range(h)]
        self.Wv=[nn.Linear(d,self.dk,bias=False) for _ in range(h)]
        self.Wo=nn.Linear(d,d)
        for i in range(h):
            self.add_module(f"Wq{i}",self.Wq[i])
            self.add_module(f"Wk{i}",self.Wk[i])
            self.add_module(f"Wv{i}",self.Wv[i])
    def forward(self,x):
        heads=[]
        for i in range(self.h):
            Q=self.Wq[i](x); K=self.Wk[i](x); V=self.Wv[i](x)
            s=Q@K.transpose(-2,-1)/self.dk**0.5
            w=F.softmax(s,-1); heads.append(w@V)
        cat=torch.cat(heads,-1)
        return self.Wo(cat)
mha_scratch=MHA(8,2); x=torch.randn(1,4,8)
out=mha_scratch(x); print(f"  MHA output: {out.shape}")

# Ex 4
print("\n=== Ex 4: Attention on Embeddings ===")
embed=nn.Embedding(10,16); x=torch.tensor([[1,5,3,7,2]])
emb=embed(x); Q=K=V=emb.squeeze(0).detach().numpy()
_,w=sdpa(Q,K,V)
WORDS=["cat","dog","sat","mat","hat"]
print("  Strongest attention pairs:")
for i in range(5):
    j=np.argmax(w[i]); print(f"    '{WORDS[i]}' attends most to '{WORDS[j]}' (w={w[i,j]:.3f})")

# Ex 5
print("\n=== Ex 5: Attention Entropy ===")
T=5; Q=np.random.randn(T,8); K=np.random.randn(T,8); V=np.random.randn(T,8)
_,w=sdpa(Q,K,V)
entropies=-np.sum(w*np.log(w+1e-9),axis=-1)
max_ent=np.log(T)
print(f"  Max possible entropy: {max_ent:.3f} (uniform)")
for i,(e,row) in enumerate(zip(entropies,w)):
    focus="focused" if e<max_ent*0.6 else "diffuse"
    print(f"    Token {i}: entropy={e:.3f} ({focus}) top={np.argmax(row)}")
