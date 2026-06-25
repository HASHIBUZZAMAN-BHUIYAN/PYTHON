# Advanced Day 32 — Solutions
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

torch.manual_seed(42); np.random.seed(42)

MEANS = [(2,2),(-2,2),(2,-2),(-2,-2)]
def sample_real(n=256, label=None):
    data,labs=[],[]
    for _ in range(n):
        l=label if label is not None else np.random.randint(4)
        mu=MEANS[l]; data.append(np.array(mu)+0.4*np.random.randn(2)); labs.append(l)
    return torch.tensor(np.array(data,dtype=np.float32)),torch.tensor(labs)

Z_DIM=8
class G_Base(nn.Module):
    def __init__(self,extra=0):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(Z_DIM+extra,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,2))
    def forward(self,z): return self.net(z)
class D_Base(nn.Module):
    def __init__(self,sig=True,extra=0):
        super().__init__()
        layers=[nn.Linear(2+extra,32),nn.LeakyReLU(0.2),nn.Linear(32,32),nn.LeakyReLU(0.2),nn.Linear(32,1)]
        if sig: layers.append(nn.Sigmoid())
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

def train_gan(G,D,opt_G,opt_D,loss_fn_d,loss_fn_g,n_epochs=300,batch=64,cond=False):
    g_l,d_l=[],[]
    for _ in range(n_epochs):
        xr,lr=sample_real(batch)
        z=torch.randn(batch,Z_DIM)
        if cond:
            lc=torch.randint(0,4,(batch,)); one_hot=torch.zeros(batch,4); one_hot.scatter_(1,lc.unsqueeze(1),1.)
            xf=G(torch.cat([z,one_hot],1))
            xr_c,_=sample_real(batch); d_in_r=torch.cat([xr_c,one_hot],1); d_in_f=torch.cat([xf.detach(),one_hot],1)
        else:
            xf=G(z); d_in_r=xr; d_in_f=xf.detach()
        ld=loss_fn_d(D,d_in_r,d_in_f); opt_D.zero_grad(); ld.backward(); opt_D.step()
        z=torch.randn(batch,Z_DIM)
        if cond:
            lc=torch.randint(0,4,(batch,)); one_hot=torch.zeros(batch,4); one_hot.scatter_(1,lc.unsqueeze(1),1.)
            xf=G(torch.cat([z,one_hot],1)); d_in=torch.cat([xf,one_hot],1)
        else:
            xf=G(z); d_in=xf
        lg=loss_fn_g(D,d_in); opt_G.zero_grad(); lg.backward(); opt_G.step()
        g_l.append(lg.item()); d_l.append(ld.item())
    return g_l,d_l

bce=nn.BCELoss()
def bce_d(D,xr,xf,b=64): return bce(D(xr),torch.ones(len(xr),1))+bce(D(xf),torch.zeros(len(xf),1))
def bce_g(D,xf): return bce(D(xf),torch.ones(len(xf),1))

# Ex 1
print("=== Ex 1: Mode Collapse ===")
G1=G_Base(); D1=D_Base()
g_l,_=train_gan(G1,D1,optim.Adam(G1.parameters(),lr=1e-5),optim.Adam(D1.parameters(),lr=1e-3),bce_d,bce_g,300)
G1.eval()
with torch.no_grad(): pts=G1(torch.randn(256,Z_DIM)).numpy()
spread=pts.std(0).mean(); print(f"  Generated spread: {spread:.3f} (real≈2.0); mode collapse if <0.5)")

# Ex 2
print("\n=== Ex 2: WGAN ===")
G2=G_Base(); D2=D_Base(sig=False)
def wgan_d(D,xr,xf):
    loss=-D(xr).mean()+D(xf).mean(); return loss
def wgan_g(D,xf): return -D(xf).mean()
def clip_d(D): [p.data.clamp_(-0.01,0.01) for p in D.parameters()]
opt_G2=optim.RMSprop(G2.parameters(),lr=5e-5); opt_D2=optim.RMSprop(D2.parameters(),lr=5e-5)
for _ in range(300):
    xr,_=sample_real(64); z=torch.randn(64,Z_DIM); xf=G2(z)
    ld=wgan_d(D2,xr,xf); opt_D2.zero_grad(); ld.backward(); opt_D2.step(); clip_d(D2)
    xf=G2(torch.randn(64,Z_DIM)); lg=wgan_g(D2,xf); opt_G2.zero_grad(); lg.backward(); opt_G2.step()
G2.eval()
with torch.no_grad(): pts_w=G2(torch.randn(256,Z_DIM)).numpy()
print(f"  WGAN spread: {pts_w.std(0).mean():.3f}")

# Ex 3
print("\n=== Ex 3: Conditional GAN ===")
G3=nn.Sequential(nn.Linear(Z_DIM+4,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,2))
D3=nn.Sequential(nn.Linear(2+4,32),nn.LeakyReLU(0.2),nn.Linear(32,32),nn.LeakyReLU(0.2),nn.Linear(32,1),nn.Sigmoid())
opt_G3=optim.Adam(G3.parameters(),lr=2e-4); opt_D3=optim.Adam(D3.parameters(),lr=2e-4)
for _ in range(400):
    bs=64; xr,lr=sample_real(bs); z=torch.randn(bs,Z_DIM)
    lc=torch.randint(0,4,(bs,)); oh=torch.zeros(bs,4); oh.scatter_(1,lc.unsqueeze(1),1.)
    xf=G3(torch.cat([z,oh],1)).detach()
    xr_r,lr_r=sample_real(bs); oh_r=torch.zeros(bs,4); oh_r.scatter_(1,lr_r.unsqueeze(1),1.)
    ld=bce(D3(torch.cat([xr_r,oh_r],1)),torch.ones(bs,1))+bce(D3(torch.cat([xf,oh],1)),torch.zeros(bs,1))
    opt_D3.zero_grad(); ld.backward(); opt_D3.step()
    z=torch.randn(bs,Z_DIM); lc=torch.randint(0,4,(bs,)); oh=torch.zeros(bs,4); oh.scatter_(1,lc.unsqueeze(1),1.)
    xf=G3(torch.cat([z,oh],1))
    lg=bce(D3(torch.cat([xf,oh],1)),torch.ones(bs,1)); opt_G3.zero_grad(); lg.backward(); opt_G3.step()
G3.eval()
print("  Conditional samples per mode:")
for i,mu in enumerate(MEANS):
    oh=torch.zeros(1,4); oh[0,i]=1.
    with torch.no_grad(): pt=G3(torch.cat([torch.randn(1,Z_DIM),oh],1)).squeeze().numpy()
    print(f"    Mode {i} (true {mu}): generated ({pt[0]:.2f},{pt[1]:.2f})")

# Ex 4 & 5
print("\n=== Ex 4+5: Diversity & FD ===")
G4=G_Base(); D4=D_Base()
train_gan(G4,D4,optim.Adam(G4.parameters(),lr=2e-4),optim.Adam(D4.parameters(),lr=2e-4),bce_d,bce_g,400)
G4.eval()
with torch.no_grad(): gen=G4(torch.randn(512,Z_DIM)).numpy()
real,_=sample_real(512); real=real.numpy()
gen_div=np.sqrt(((gen[None]-gen[:,None])**2).sum(-1)).mean()
real_div=np.sqrt(((real[None]-real[:,None])**2).sum(-1)).mean()
print(f"  Diversity: real={real_div:.2f}  generated={gen_div:.2f}")
mu_r,sig_r=real.mean(0),np.cov(real.T)
mu_g,sig_g=gen.mean(0),np.cov(gen.T)
fd=np.sum((mu_r-mu_g)**2)+np.trace(sig_r+sig_g-2*sqrtm(sig_r@sig_g).real)
print(f"  Frechet Distance: {fd:.4f}  (0=perfect match)")
