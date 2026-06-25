# Advanced Day 12 — Solutions
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from sklearn.datasets import load_digits; from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
np.random.seed(42); torch.manual_seed(42)

digits=load_digits(); X=digits.data.astype(np.float32)/16; y=digits.target
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
Xtt=torch.FloatTensor(Xtr).view(-1,1,8,8); ytt=torch.LongTensor(ytr)
Xtet=torch.FloatTensor(Xte).view(-1,1,8,8); ytet=torch.LongTensor(yte)

# Exercise 1
def add_noise(img,s=0.05):     return np.clip(img+np.random.normal(0,s,img.shape),0,1)
def bright_jitter(img,r=(0.8,1.2)): return np.clip(img*(r[0]+np.random.rand()*(r[1]-r[0])),0,1)
def h_flip(img):                return img[:,::-1].copy()
def crop_pad(img,c=1):          return np.pad(img[c:-c,c:-c],c,mode="edge")
def rot90(img):                 return np.rot90(img).copy()
sample=Xtr[0].reshape(8,8)
fig,axes=plt.subplots(1,6,figsize=(12,2))
for ax,(t,fn) in zip(axes,[("orig",None),("noise",add_noise),("bright",bright_jitter),("flip",h_flip),("crop",crop_pad),("rot90",rot90)]):
    ax.imshow(fn(sample) if fn else sample,cmap="gray"); ax.set_title(t); ax.axis("off")
plt.tight_layout(); plt.savefig("aug_demo.png",dpi=72); plt.close()

# Exercise 2 + 3
class CNN2(nn.Module):
    def __init__(self,bn=True,drop_pos="fc"):
        super().__init__()
        self.c1=nn.Conv2d(1,16,3,padding=1); self.c2=nn.Conv2d(16,32,3,padding=1)
        self.b1=nn.BatchNorm2d(16) if bn else nn.Identity(); self.b2=nn.BatchNorm2d(32) if bn else nn.Identity()
        self.pool=nn.MaxPool2d(2); self.fc1=nn.Linear(32*2*2,64); self.fc2=nn.Linear(64,10); self.drop=nn.Dropout(0.3); self.drop_pos=drop_pos
    def forward(self,x):
        x=self.pool(F.relu(self.b1(self.c1(x)))); x=self.pool(F.relu(self.b2(self.c2(x)))); x=x.view(x.size(0),-1)
        if self.drop_pos=="fc": x=self.drop(F.relu(self.fc1(x)))
        else: x=F.relu(self.fc1(x))
        return self.fc2(x)

def run(m,epochs=15,lr=1e-3):
    opt=optim.Adam(m.parameters(),lr); crit=nn.CrossEntropyLoss(); hist=[]
    for ep in range(epochs):
        for xb,yb in DataLoader(TensorDataset(Xtt,ytt),32,shuffle=True):
            opt.zero_grad(); l=crit(m(xb),yb); l.backward(); opt.step()
        with torch.no_grad(): hist.append((m(Xtet).argmax(1)==ytet).float().mean().item())
    return hist

results={n:run(CNN2(bn,dp)) for n,bn,dp in [("NoBN",False,"fc"),("BN",True,"fc"),("BN+DropConv",True,"conv")]}
fig,ax=plt.subplots(figsize=(8,4))
for n,h in results.items(): ax.plot(h,label=n)
ax.legend(); ax.set_title("BN / Dropout placement"); plt.savefig("bn_drop.png",dpi=72); plt.close()

# Exercise 5 — Gradient clipping
m5=CNN2(); opt5=optim.SGD(m5.parameters(),lr=0.1); crit5=nn.CrossEntropyLoss(); ls_clip=[]; ls_no=[]
m5_no=CNN2(); opt_no=optim.SGD(m5_no.parameters(),lr=0.1)
for ep in range(10):
    for xb,yb in DataLoader(TensorDataset(Xtt,ytt),32,shuffle=True):
        opt5.zero_grad(); l=crit5(m5(xb),yb); l.backward()
        torch.nn.utils.clip_grad_norm_(m5.parameters(),1.0); opt5.step()
        opt_no.zero_grad(); l2=crit5(m5_no(xb),yb); l2.backward(); opt_no.step()
    with torch.no_grad():
        ls_clip.append((m5(Xtet).argmax(1)==ytet).float().mean().item())
        ls_no.append((m5_no(Xtet).argmax(1)==ytet).float().mean().item())
print(f"With clip: {ls_clip[-1]:.3f}  Without clip: {ls_no[-1]:.3f}")
print("Saved aug_demo.png, bn_drop.png")
