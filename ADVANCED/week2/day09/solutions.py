# Advanced Day 09 — Solutions
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_moons, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
np.random.seed(42); torch.manual_seed(42)

# Exercise 1
A = torch.randn(5,5); print("Inverse:\n", torch.linalg.inv(A).round(decimals=3))
T1 = torch.randn(4,3,2); T2 = torch.randn(4,2,5)
print("bmm:", torch.bmm(T1,T2).shape)
stacked = torch.stack([torch.randn(2,2) for _ in range(4)]); print("Stacked:", stacked.shape)

# Exercise 2
class HighwayBlock(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.H = nn.Sequential(nn.Linear(n,n),nn.ReLU())
        self.T = nn.Sequential(nn.Linear(n,n),nn.Sigmoid())
    def forward(self,x): t=self.T(x); return self.H(x)*t + x*(1-t)

net = nn.Sequential(nn.Linear(10,64),*[HighwayBlock(64) for _ in range(3)],nn.Linear(64,1))
print("Highway params:", sum(p.numel() for p in net.parameters()))
print("Test:", net(torch.randn(16,10)).shape)

# Exercise 3
X_m,y_m = make_moons(400,noise=0.2,random_state=42)
Xtr,Xte,ytr,yte = train_test_split(X_m,y_m,test_size=0.2)
sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
Xt=torch.FloatTensor(Xtr); yt=torch.FloatTensor(ytr).unsqueeze(1)
fig,ax=plt.subplots(figsize=(8,4))
for name,opt_fn in [("SGD",optim.SGD),("Adam",optim.Adam),("RMSprop",optim.RMSprop)]:
    tiny=nn.Sequential(nn.Linear(2,16),nn.ReLU(),nn.Linear(16,1),nn.Sigmoid())
    opt=opt_fn(tiny.parameters(),lr=0.05 if name=="SGD" else 0.01)
    ls=[]
    for _ in range(200):
        opt.zero_grad(); out=tiny(Xt); loss=nn.BCELoss()(out,yt); loss.backward(); opt.step(); ls.append(loss.item())
    ax.plot(ls,label=name)
ax.legend(); ax.set_title("Optimizer comparison"); plt.savefig("optimizers.png",dpi=72); plt.close()

# Exercise 4
m=nn.Sequential(nn.Linear(2,8),nn.ReLU(),nn.Linear(8,1))
torch.save(m.state_dict(),"tiny_model.pt"); m2=nn.Sequential(nn.Linear(2,8),nn.ReLU(),nn.Linear(8,1))
m2.load_state_dict(torch.load("tiny_model.pt")); x=torch.randn(4,2)
print("Same output:", torch.allclose(m(x),m2(x)))

# Exercise 5
import os; os.remove("tiny_model.pt")
digits=load_digits(); sc2=StandardScaler()
X2=sc2.fit_transform(digits.data).astype(np.float32)
Xtr2,Xte2,ytr2,yte2=train_test_split(X2,digits.target,test_size=0.2,random_state=42)
loader2=DataLoader(TensorDataset(torch.FloatTensor(Xtr2),torch.LongTensor(ytr2)),batch_size=32,shuffle=True)
for use_bn in [False,True]:
    layers=[nn.Linear(64,128),(nn.BatchNorm1d(128) if use_bn else nn.Identity()),nn.ReLU(),nn.Linear(128,10)]
    m3=nn.Sequential(*layers); opt3=optim.Adam(m3.parameters(),lr=1e-3); ls=[]
    for ep in range(10):
        el=0
        for xb,yb in loader2:
            opt3.zero_grad(); o=m3(xb); l=nn.CrossEntropyLoss()(o,yb); l.backward(); opt3.step(); el+=l.item()
        ls.append(el/len(loader2))
    print(f"BatchNorm={use_bn}: final loss={ls[-1]:.4f}")
print("Saved optimizers.png")
