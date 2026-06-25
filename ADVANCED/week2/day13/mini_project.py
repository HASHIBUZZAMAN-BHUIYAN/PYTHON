# Advanced Day 13 Mini-Project — Shapes Classifier with Transfer Learning Report
# Generates a full report comparing pretrained vs scratch on a synthetic shapes dataset.
# ~500 MB RAM, ~10-15 min on CPU — close other apps first.

import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models

torch.manual_seed(42); np.random.seed(42)

CLASS_NAMES = ["Circle", "Square", "Triangle"]
N_PER = 200   # 200 per class — faster; increase for better accuracy

def make_shape(cls, size=48):
    img = np.ones((size,size,3),dtype=np.float32)*0.9
    c, r = size//2, size//4
    noise_lvl = 0.06
    if cls==0:
        for y in range(size):
            for x in range(size):
                if (x-c)**2+(y-c)**2 <= r**2: img[y,x]=[0.2,0.4,0.8]
    elif cls==1:
        img[c-r:c+r, c-r:c+r] = [0.8,0.2,0.2]
    else:
        for y in range(size):
            hw=max(0,int((y-(c-r))*r/r))
            if c-r<=y<=c+r: img[y,max(0,c-hw):min(size,c+hw)]=[0.2,0.8,0.2]
    return np.clip(img+np.random.normal(0,noise_lvl,img.shape).astype(np.float32),0,1)

print("Generating dataset ...")
X=[make_shape(cls) for cls in range(3) for _ in range(N_PER)]
y=[cls           for cls in range(3) for _ in range(N_PER)]
X,y=np.array(X),np.array(y)
idx=np.random.permutation(len(X)); X,y=X[idx],y[idx]
sp=int(0.8*len(X)); Xtr,Xte,ytr,yte=X[:sp],X[sp:],y[:sp],y[sp:]

mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
std =torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
to_t=lambda X: (F.interpolate(torch.FloatTensor(X).permute(0,3,1,2),96)-mean)/std
Xtt=to_t(Xtr); Xtet=to_t(Xte); ytt=torch.LongTensor(ytr); ytet=torch.LongTensor(yte)
loader=DataLoader(TensorDataset(Xtt,ytt),32,shuffle=True)

def train_model(m, epochs, lr, label):
    opt=optim.Adam(filter(lambda p:p.requires_grad,m.parameters()),lr)
    crit=nn.CrossEntropyLoss(); hist=[]
    for ep in range(epochs):
        m.train()
        for xb,yb in loader:
            opt.zero_grad(); l=crit(m(xb),yb); l.backward(); opt.step()
        m.eval()
        with torch.no_grad(): hist.append((m(Xtet).argmax(1)==ytet).float().mean().item())
    print(f"{label:<35}: {hist[-1]:.4f}")
    return hist

EPOCHS=15
# Model 1: MobileNetV2 (feature extraction)
mv2=models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
for p in mv2.parameters(): p.requires_grad=False
mv2.classifier=nn.Sequential(nn.Dropout(0.2),nn.Linear(mv2.classifier[1].in_features,3))
h_mv2=train_model(mv2,EPOCHS,1e-3,"MobileNetV2 (frozen backbone)")

# Model 2: Scratch CNN
scratch=nn.Sequential(
    nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
    nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
    nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),nn.Dropout(0.3),nn.Linear(128,3)
)
h_scratch=train_model(scratch,EPOCHS,1e-3,"CNN from scratch")

# ─── Report ──────────────────────────────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(12,5))
fig.suptitle("Transfer Learning vs From Scratch — Shapes Classification")
axes[0].plot(h_mv2,label=f"MobileNetV2 ({h_mv2[-1]:.3f})")
axes[0].plot(h_scratch,label=f"Scratch ({h_scratch[-1]:.3f})",linestyle="--")
axes[0].set_title("Test Accuracy"); axes[0].legend(); axes[0].set_xlabel("Epoch")

samples=[[make_shape(c) for c in range(3)] for _ in range(3)]
for i,row in enumerate(samples):
    for j,img in enumerate(row):
        axes[1].imshow(img,extent=[j,j+1,3-i-1,3-i]); axes[1].text(j+0.5,3-i-1.15,CLASS_NAMES[j],ha="center",fontsize=8)
axes[1].set_xlim(0,3); axes[1].set_ylim(-0.3,3); axes[1].axis("off"); axes[1].set_title("Sample Training Images")

plt.tight_layout(); plt.savefig("transfer_report.png",dpi=80)
print("\nSaved transfer_report.png")
plt.show()
