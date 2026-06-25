# Advanced Day 31 — Solutions
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

torch.manual_seed(42); np.random.seed(42)

def make_data(n=300):
    t=np.linspace(0,4*np.pi,n); x=t*np.cos(t); y=t*np.sin(t)
    xy=np.column_stack([x,y]); xy=(xy-xy.mean(0))/(xy.std(0)+1e-8)
    proj=np.random.randn(2,8)
    return (xy@proj + 0.1*np.random.randn(n,8)).astype(np.float32), xy

X_np,xy=make_data(); X=torch.tensor(X_np)

def make_ae(in_dim,latent,deep=False):
    if deep:
        enc=nn.Sequential(nn.Linear(in_dim,16),nn.ReLU(),nn.Linear(16,8),nn.ReLU(),nn.Linear(8,latent))
        dec=nn.Sequential(nn.Linear(latent,8),nn.ReLU(),nn.Linear(8,16),nn.ReLU(),nn.Linear(16,in_dim))
    else:
        enc=nn.Sequential(nn.Linear(in_dim,16),nn.ReLU(),nn.Linear(16,latent))
        dec=nn.Sequential(nn.Linear(latent,16),nn.ReLU(),nn.Linear(16,in_dim))
    class AE(nn.Module):
        def forward(s,x): z=enc(x); return dec(z),z
    m=AE(); m.encoder=enc; m.decoder=dec; return m

def train(ae,X,epochs=300,lr=1e-3,noise=0.0):
    opt=optim.Adam(ae.parameters(),lr=lr); crit=nn.MSELoss()
    for _ in range(epochs):
        Xin=X+noise*torch.randn_like(X) if noise>0 else X
        Xh,_=ae(Xin); loss=crit(Xh,X)
        opt.zero_grad(); loss.backward(); opt.step()
    ae.eval()
    with torch.no_grad(): Xh,_=ae(X)
    return crit(Xh,X).item()

# Ex 1
print("=== Ex 1: Latent Dim Effect ===")
losses=[]; dims=[1,2,4,8]
for d in dims:
    ae=make_ae(8,d); l=train(ae,X); losses.append(l)
    print(f"  latent_dim={d}: final_loss={l:.5f}")
fig,ax=plt.subplots(figsize=(5,3))
ax.bar([str(d) for d in dims],losses,color="steelblue",alpha=0.8)
ax.set_xlabel("Latent Dim"); ax.set_ylabel("MSE"); ax.set_title("Loss vs Latent Dim")
plt.tight_layout(); plt.savefig("latent_dim.png",dpi=80); plt.close(); print("  Saved latent_dim.png")

# Ex 2
print("\n=== Ex 2: Deeper Encoder ===")
shallow_l=train(make_ae(8,2,deep=False),X)
deep_l   =train(make_ae(8,2,deep=True),X)
print(f"  Shallow (8→16→2): {shallow_l:.5f}")
print(f"  Deep    (8→16→8→2): {deep_l:.5f}")

# Ex 3
print("\n=== Ex 3: Denoising at different noise levels ===")
noise_levels=[0.1,0.5,1.0,2.0]; improvements=[]
for noise in noise_levels:
    ae=make_ae(8,2)
    opt=optim.Adam(ae.parameters(),lr=1e-3); crit=nn.MSELoss()
    for _ in range(300):
        Xn=X+noise*torch.randn_like(X); Xh,_=ae(Xn); loss=crit(Xh,X)
        opt.zero_grad(); loss.backward(); opt.step()
    ae.eval()
    Xn_test=X+noise*torch.randn_like(X)
    with torch.no_grad(): Xd,_=ae(Xn_test)
    noisy_mse   =((Xn_test-X)**2).mean().item()
    denoised_mse=((Xd-X)**2).mean().item()
    imp=(noisy_mse-denoised_mse)/noisy_mse
    improvements.append(imp)
    print(f"  noise={noise:.1f}: noisy_mse={noisy_mse:.4f} denoised={denoised_mse:.4f} improvement={imp:.1%}")
fig,ax=plt.subplots(figsize=(5,3))
ax.plot(noise_levels,[i*100 for i in improvements],"o-",color="tomato")
ax.set_xlabel("Noise Level"); ax.set_ylabel("Improvement %"); ax.set_title("Denoising Improvement")
plt.tight_layout(); plt.savefig("denoising.png",dpi=80); plt.close(); print("  Saved denoising.png")

# Ex 4
print("\n=== Ex 4: Anomaly Threshold Sweep ===")
ae4=make_ae(8,2); train(ae4,X[:250],epochs=500)
ae4.eval()
X_anom=3.0+0.3*torch.randn(20,8)
with torch.no_grad():
    n_err=((ae4(X[:250])[0]-X[:250])**2).mean(dim=1).numpy()
    a_err=((ae4(X_anom)[0]-X_anom)**2).mean(dim=1).numpy()
thresholds=np.percentile(n_err,np.arange(50,100,5))
tprs=[]; fprs=[]
for t in thresholds:
    tprs.append(np.mean(a_err>t)); fprs.append(np.mean(n_err>t))
print(f"  FPR range: {min(fprs):.2f}–{max(fprs):.2f}  TPR range: {min(tprs):.2f}–{max(tprs):.2f}")

# Ex 5
print("\n=== Ex 5: PCA vs AE ===")
ae5=make_ae(8,2); train(ae5,X,epochs=300)
with torch.no_grad(): _,Z=ae5(X)
Z_ae=Z.numpy()
Z_pca=PCA(n_components=2).fit_transform(X_np)
fig,axes=plt.subplots(1,2,figsize=(9,4))
axes[0].scatter(Z_pca[:,0],Z_pca[:,1],c=np.arange(len(X_np)),cmap="viridis",s=15,alpha=0.7)
axes[0].set_title("PCA 2D")
axes[1].scatter(Z_ae[:,0],Z_ae[:,1],c=np.arange(len(X_np)),cmap="viridis",s=15,alpha=0.7)
axes[1].set_title("AE Latent 2D")
plt.suptitle("PCA vs AE: 2D Projection"); plt.tight_layout()
plt.savefig("pca_vs_ae.png",dpi=80); plt.close(); print("  Saved pca_vs_ae.png")
