"""
Project: Tiny 8×8 Image GAN
Teaches: training a GAN to generate simple 8×8 pattern images (circles/lines/noise),
         visualizing a grid of generated images.
~60 MB RAM, ~15s on CPU
"""
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42); np.random.seed(42)

IMG_DIM = 64  # 8×8 flattened

def make_training_images(n_each=200):
    imgs = []
    # 1. Circle patterns
    for _ in range(n_each):
        img=np.zeros((8,8)); cx=cy=3.5; r=np.random.uniform(1.2,2.8)
        for i in range(8):
            for j in range(8):
                if abs(np.sqrt((i-cx)**2+(j-cy)**2)-r)<0.9: img[i,j]=1.0
        imgs.append(img.flatten())
    # 2. Horizontal stripe patterns
    for _ in range(n_each):
        img=np.zeros((8,8))
        rows=np.random.choice(8,np.random.randint(2,5),replace=False)
        img[rows,:]=1.0; imgs.append(img.flatten())
    # 3. Diagonal patterns
    for _ in range(n_each):
        img=np.zeros((8,8)); offset=np.random.randint(-4,5)
        for i in range(8):
            j=(i+offset)%8; img[i,j]=1.0
        imgs.append(img.flatten())
    X=np.array(imgs,dtype=np.float32); idx=np.random.permutation(len(X))
    return torch.tensor(X[idx])

X_train=make_training_images(200)
print(f"Training images: {X_train.shape}")

Z_DIM=16
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(Z_DIM,32),nn.BatchNorm1d(32),nn.ReLU(),
            nn.Linear(32,64),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Linear(64,IMG_DIM),nn.Sigmoid())
    def forward(self,z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(IMG_DIM,64),nn.LeakyReLU(0.2),nn.Dropout(0.3),
            nn.Linear(64,32),nn.LeakyReLU(0.2),
            nn.Linear(32,1),nn.Sigmoid())
    def forward(self,x): return self.net(x)

G=Generator(); D=Discriminator()
opt_G=optim.Adam(G.parameters(),lr=2e-4,betas=(0.5,0.999))
opt_D=optim.Adam(D.parameters(),lr=2e-4,betas=(0.5,0.999))
bce=nn.BCELoss(); BATCH=64; EPOCHS=600

g_hist,d_hist=[],[]
print("Training image GAN ...")
for epoch in range(EPOCHS):
    idx=torch.randperm(len(X_train))[:BATCH]
    xr=X_train[idx]; z=torch.randn(BATCH,Z_DIM); xf=G(z).detach()
    ld=bce(D(xr),torch.ones(BATCH,1))+bce(D(xf),torch.zeros(BATCH,1))
    opt_D.zero_grad(); ld.backward(); opt_D.step()

    z=torch.randn(BATCH,Z_DIM); xf=G(z)
    lg=bce(D(xf),torch.ones(BATCH,1))
    opt_G.zero_grad(); lg.backward(); opt_G.step()
    g_hist.append(lg.item()); d_hist.append(ld.item())
    if (epoch+1)%200==0: print(f"  Epoch {epoch+1}  G={lg.item():.3f}  D={ld.item():.3f}")

G.eval()
with torch.no_grad(): samples=G(torch.randn(16,Z_DIM)).numpy().reshape(16,8,8)

fig,axes=plt.subplots(4,6,figsize=(12,8))
# Real samples
for i in range(8):
    axes[i//4,i%4+2 if i<4 else i%4-2].imshow(X_train[i].numpy().reshape(8,8),cmap="binary",vmin=0,vmax=1)
# Generated
for i in range(8):
    axes[(i+8)//6,(i+8)%6].imshow(samples[i],cmap="binary",vmin=0,vmax=1)
for ax in axes.flat: ax.axis("off")
# Loss plot
ax_loss=fig.add_axes([0.7,0.05,0.28,0.35])
ax_loss.plot(g_hist,label="G",alpha=0.7,color="tomato",linewidth=0.8)
ax_loss.plot(d_hist,label="D",alpha=0.7,color="steelblue",linewidth=0.8)
ax_loss.set_title("Losses",fontsize=8); ax_loss.legend(fontsize=7)
plt.suptitle("8×8 Image GAN: Real (left) vs Generated (right)",fontsize=10)
plt.savefig("image_gan.png",dpi=85,bbox_inches="tight"); plt.close(); print("Saved image_gan.png")
