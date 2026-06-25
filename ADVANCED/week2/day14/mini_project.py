# Advanced Day 14 Mini-Project — Synthetic Shape Classifier: Full Report
# ~400 MB RAM, ~10-15 min on CPU
# Covers: dataset generation, CNN design, training, evaluation, visualization

import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

torch.manual_seed(42); np.random.seed(42)

CLASS_NAMES = ["Circle","Square","Triangle","Cross","Diamond"]
N_PER_CLASS = 150  # 5 classes × 150 = 750 train
IMG_SIZE    = 32
EPOCHS      = 20
BATCH       = 32
LR          = 5e-4

def make_image(cls, size=IMG_SIZE, noise=0.06):
    img = np.ones((size,size,3),dtype=np.float32)*0.85
    c, r = size//2, size//4
    if cls==0:   # Circle
        for y in range(size):
            for x in range(size):
                if (x-c)**2+(y-c)**2<=r**2: img[y,x]=[0.1,0.4,0.9]
    elif cls==1: # Square
        img[c-r:c+r, c-r:c+r]=[0.9,0.1,0.1]
    elif cls==2: # Triangle
        for y in range(size):
            hw=max(0,int((y-(c-r))*r/(2*r+1)))
            if c-r<=y<=c+r: img[y,max(0,c-hw):min(size,c+hw)]=[0.1,0.8,0.1]
    elif cls==3: # Cross
        img[c-1:c+2,c-r:c+r]=[0.8,0.5,0.1]; img[c-r:c+r,c-1:c+2]=[0.8,0.5,0.1]
    elif cls==4: # Diamond
        for y in range(size):
            hw=r-abs(y-c)
            if hw>0: img[y,max(0,c-hw):min(size,c+hw)]=[0.6,0.1,0.8]
    img += np.random.normal(0,noise,img.shape).astype(np.float32)
    return np.clip(img,0,1)

# Dataset
X=[make_image(c) for c in range(5) for _ in range(N_PER_CLASS)]
y=[c for c in range(5) for _ in range(N_PER_CLASS)]
X,y=np.array(X),np.array(y)
idx=np.random.permutation(len(X)); X,y=X[idx],y[idx]
sp=int(0.8*len(X)); Xtr,Xte,ytr,yte=X[:sp],X[sp:],y[:sp],y[sp:]

def to_t(X,y):
    xt=torch.FloatTensor(X).permute(0,3,1,2)
    mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std =torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    return (xt-mean)/std, torch.LongTensor(y)

Xtt,ytt=to_t(Xtr,ytr); Xtet,ytet=to_t(Xte,yte)
loader=DataLoader(TensorDataset(Xtt,ytt),BATCH,shuffle=True)

class ShapeCNN(nn.Module):
    def __init__(self,n_classes=5):
        super().__init__()
        def block(ic,oc,k=3,p=1):
            return nn.Sequential(nn.Conv2d(ic,oc,k,padding=p),nn.BatchNorm2d(oc),nn.ReLU(),nn.MaxPool2d(2))
        self.features=nn.Sequential(block(3,32),block(32,64),block(64,128))
        self.clf=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(0.3),nn.Linear(128,n_classes))
    def forward(self,x): return self.clf(self.features(x))

model=ShapeCNN(5)
opt=optim.Adam(model.parameters(),LR,weight_decay=1e-4)
sched=optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
crit=nn.CrossEntropyLoss()

best_acc,best_state=0.,None; tr_hist,te_hist=[],[]
print(f"Training {EPOCHS} epochs on {N_PER_CLASS*5} samples ({5} classes) ...")
for ep in range(1,EPOCHS+1):
    model.train(); tl,tc,tt=0.,0,0
    for xb,yb in loader:
        opt.zero_grad(); out=model(xb); l=crit(out,yb); l.backward(); opt.step()
        tl+=l.item()*len(yb); tc+=(out.argmax(1)==yb).sum().item(); tt+=len(yb)
    sched.step()
    model.eval()
    with torch.no_grad(): te_acc=(model(Xtet).argmax(1)==ytet).float().mean().item()
    tr_hist.append(tc/tt); te_hist.append(te_acc)
    if te_acc>best_acc: best_acc=te_acc; best_state=model.state_dict().copy()
    print(f"Ep {ep:2d}: train={tc/tt:.4f} test={te_acc:.4f}")

model.load_state_dict(best_state)
model.eval()
with torch.no_grad(): preds=model(Xtet).argmax(1).numpy()

print(f"\nBest test accuracy: {best_acc:.4f}")
print(classification_report(yte,preds,target_names=CLASS_NAMES))

# ─── Report ──────────────────────────────────────────────────────────────────
fig=plt.figure(figsize=(15,10)); fig.suptitle("Day 14 Capstone: Shape Classifier Report",fontsize=14)
ax1=fig.add_subplot(2,3,1); ax2=fig.add_subplot(2,3,2); ax3=fig.add_subplot(2,3,3)
ax4=fig.add_subplot(2,3,4); ax5=fig.add_subplot(2,3,5); ax6=fig.add_subplot(2,3,6)

ax1.plot(tr_hist,label="Train"); ax1.plot(te_hist,label="Test")
ax1.axhline(best_acc,color="green",linestyle="--",label=f"Best={best_acc:.3f}"); ax1.legend(fontsize=8)
ax1.set_title("Train vs Test Accuracy")

cm=confusion_matrix(yte,preds)
ax2.imshow(cm,cmap="Blues"); ax2.set_title("Confusion Matrix")
ax2.set_xticks(range(5)); ax2.set_xticklabels(CLASS_NAMES,rotation=40,fontsize=7)
ax2.set_yticks(range(5)); ax2.set_yticklabels(CLASS_NAMES,fontsize=7)
for i in range(5):
    for j in range(5): ax2.text(j,i,cm[i,j],ha="center",va="center",fontsize=8,color="white" if cm[i,j]>cm.max()*0.6 else "black")

acc_per_class=cm.diagonal()/cm.sum(axis=1)
ax3.bar(CLASS_NAMES,acc_per_class,color="steelblue"); ax3.set_title("Per-class Accuracy")
ax3.set_ylim(0,1.1); ax3.tick_params(axis="x",rotation=30)
for i,(name,acc) in enumerate(zip(CLASS_NAMES,acc_per_class)): ax3.text(i,acc+0.02,f"{acc:.2f}",ha="center",fontsize=8)

for ax,cls in zip([ax4,ax5,ax6],[0,1,2]):
    imgs=[make_image(cls) for _ in range(5)]
    strip=np.concatenate(imgs,axis=1)
    ax.imshow(strip); ax.set_title(f"Sample: {CLASS_NAMES[cls]}"); ax.axis("off")

plt.tight_layout(); plt.savefig("day14_report.png",dpi=80,bbox_inches="tight")
print("Saved day14_report.png"); plt.show()
