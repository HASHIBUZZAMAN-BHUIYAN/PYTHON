"""
Project: 2D Distribution GAN
Teaches: training a GAN to match a complex 2D distribution (ring + blobs),
         visualizing the progression of generated samples over training.
~40 MB RAM, ~5s on CPU
"""
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)

# ─── Target: ring (inner) + 4 corner blobs ────────────────────────────────────
def sample_real(n=256):
    n_ring = n // 2
    n_blobs= n - n_ring
    # Ring
    theta = np.random.uniform(0, 2*np.pi, n_ring)
    r     = np.random.normal(3.0, 0.3, n_ring)
    ring  = np.column_stack([r*np.cos(theta), r*np.sin(theta)])
    # Blobs
    centers=[(5,5),(-5,5),(5,-5),(-5,-5)]
    blobs=[]; per=n_blobs//4
    for c in centers:
        blobs.append(np.array(c) + 0.5*np.random.randn(per,2))
    blobs = np.vstack(blobs)
    all_pts = np.vstack([ring, blobs]).astype(np.float32)
    idx = np.random.permutation(len(all_pts))
    return torch.tensor(all_pts[idx])

Z_DIM = 16
class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(Z_DIM,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2))
    def forward(self,z): return self.net(z)

class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(2,64),nn.LeakyReLU(0.2),nn.Dropout(0.3),
            nn.Linear(64,64),nn.LeakyReLU(0.2),
            nn.Linear(64,1),nn.Sigmoid())
    def forward(self,x): return self.net(x)

gen=G(); dis=D()
opt_G=optim.Adam(gen.parameters(),lr=1e-4,betas=(0.5,0.999))
opt_D=optim.Adam(dis.parameters(),lr=2e-4,betas=(0.5,0.999))
bce=nn.BCELoss()

g_loss_hist,d_loss_hist=[],[]
snapshots=[]
EPOCHS=600; BATCH=128

print("Training GAN on ring + blobs distribution ...")
for epoch in range(EPOCHS):
    xr=sample_real(BATCH)
    z=torch.randn(BATCH,Z_DIM); xf=gen(z).detach()
    ld=bce(dis(xr),torch.ones(BATCH,1))+bce(dis(xf),torch.zeros(BATCH,1))
    opt_D.zero_grad(); ld.backward(); opt_D.step()

    z=torch.randn(BATCH,Z_DIM); xf=gen(z)
    lg=bce(dis(xf),torch.ones(BATCH,1))
    opt_G.zero_grad(); lg.backward(); opt_G.step()

    g_loss_hist.append(lg.item()); d_loss_hist.append(ld.item())
    if epoch in [0,100,300,599]:
        gen.eval()
        with torch.no_grad(): pts=gen(torch.randn(256,Z_DIM)).numpy()
        snapshots.append((epoch+1, pts.copy()))
        gen.train()
    if (epoch+1)%200==0: print(f"  Epoch {epoch+1}  G={lg.item():.3f}  D={ld.item():.3f}")

# ─── Visualize ────────────────────────────────────────────────────────────────
real_pts=sample_real(512).numpy()
fig,axes=plt.subplots(2,3,figsize=(13,8))
axes[0,0].scatter(real_pts[:,0],real_pts[:,1],s=6,alpha=0.5,c="steelblue")
axes[0,0].set_title("Real Distribution"); axes[0,0].set_xlim(-8,8); axes[0,0].set_ylim(-8,8)

for i,(ep,pts) in enumerate(snapshots):
    r,c=divmod(i+1,3); ax=axes[r,c]
    ax.scatter(pts[:,0],pts[:,1],s=6,alpha=0.5,c="tomato")
    ax.set_title(f"Generated epoch {ep}"); ax.set_xlim(-8,8); ax.set_ylim(-8,8)

axes[1,2].plot(g_loss_hist,label="G",alpha=0.7,color="tomato")
axes[1,2].plot(d_loss_hist,label="D",alpha=0.7,color="steelblue")
axes[1,2].set_title("Losses"); axes[1,2].set_xlabel("Step"); axes[1,2].legend()

plt.suptitle("2D GAN: Ring + Blobs", fontsize=11)
plt.tight_layout(); plt.savefig("2d_gan.png",dpi=85); plt.close(); print("Saved 2d_gan.png")
