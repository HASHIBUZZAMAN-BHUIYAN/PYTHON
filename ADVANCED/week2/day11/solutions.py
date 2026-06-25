# Advanced Day 11 — Solutions
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
np.random.seed(42); torch.manual_seed(42)

# Exercise 1
from scipy.ndimage import convolve
img = np.zeros((10,10)); img[3:7,3:7]=1; img[1,1]=0.5; img[8,8]=0.8
kernels = {
    "Identity": np.array([[0,0,0],[0,1,0],[0,0,0]],float),
    "Sharpen":  np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],float),
    "Blur":     np.array([[1,2,1],[2,4,2],[1,2,1]],float)/16,
    "Sobel_Y":  np.array([[-1,-2,-1],[0,0,0],[1,2,1]],float),
}
fig,axes=plt.subplots(1,5,figsize=(14,3))
axes[0].imshow(img,cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")
for ax,(name,k) in zip(axes[1:],kernels.items()):
    ax.imshow(convolve(img,k),cmap="gray"); ax.set_title(name); ax.axis("off")
plt.tight_layout(); plt.savefig("conv_kernels.png",dpi=72); plt.close()

# Exercise 2
# After 3x3 (s=1,p=0): RF=3. After another 3x3: RF=5.
# 32×32 → conv(3x3,p=0) → 30×30 → pool(2,2) → 15×15 → conv(3x3,p=0) → 13×13 → pool(2,2) → 6×6
print("Receptive field after 2×(3x3 conv, no pad): 5")
x=torch.zeros(1,1,32,32)
m=nn.Sequential(nn.Conv2d(1,4,3),nn.MaxPool2d(2),nn.Conv2d(4,8,3),nn.MaxPool2d(2))
print("Spatial size after 2 conv+pool:", m(x).shape)

# Exercise 3
ic,oc,k=32,64,3
standard = ic*oc*k*k; dw = ic*k*k; pw=ic*oc; sep=dw+pw
print(f"Standard: {standard}, Depthwise-sep: {sep}, Ratio: {sep/standard:.2f}x")
class DSConv(nn.Module):
    def __init__(self,ic,oc,k=3,p=1):
        super().__init__()
        self.dw=nn.Conv2d(ic,ic,k,padding=p,groups=ic)
        self.pw=nn.Conv2d(ic,oc,1)
    def forward(self,x): return self.pw(self.dw(x))
print("DS output:", DSConv(32,64)(torch.zeros(1,32,8,8)).shape)

# Exercise 5 — Filter visualization (train SmallCNN first)
from sklearn.datasets import load_digits; from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
digits=load_digits(); X=digits.data.astype(np.float32)/16; y=digits.target
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
Xtt=torch.FloatTensor(Xtr).view(-1,1,8,8); ytt=torch.LongTensor(ytr)
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__(); self.conv1=nn.Conv2d(1,8,3,padding=1); self.pool=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(8,16,3,padding=1); self.fc1=nn.Linear(16*2*2,64); self.fc2=nn.Linear(64,10)
    def forward(self,x): x=self.pool(F.relu(self.conv1(x))); x=self.pool(F.relu(self.conv2(x))); x=x.view(x.size(0),-1); return self.fc2(F.relu(self.fc1(x)))
cnn=SmallCNN(); opt=torch.optim.Adam(cnn.parameters(),1e-3); crit=nn.CrossEntropyLoss()
for _ in range(10):
    for xb,yb in DataLoader(TensorDataset(Xtt,ytt),32,shuffle=True):
        opt.zero_grad(); l=crit(cnn(xb),yb); l.backward(); opt.step()
filters=cnn.conv1.weight.detach().numpy()  # (8,1,3,3)
fig,axes=plt.subplots(2,4,figsize=(8,5))
for i,ax in enumerate(axes.flat): ax.imshow(filters[i,0],cmap="RdBu"); ax.set_title(f"F{i}"); ax.axis("off")
plt.suptitle("Learned Conv1 Filters"); plt.tight_layout(); plt.savefig("filters.png",dpi=72); plt.close()
print("Saved conv_kernels.png, filters.png")
