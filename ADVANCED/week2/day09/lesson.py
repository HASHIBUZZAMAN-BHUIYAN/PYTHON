# Advanced Day 09 — Intro to PyTorch (CPU)
# ~300 MB RAM, ~1-2 min on CPU

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42); np.random.seed(42)
print(f"PyTorch version: {torch.__version__}")
print(f"Device: CPU (no CUDA needed)")

# ─── 1. TENSORS ──────────────────────────────────────────────────────────────
print("\n=== 1. Tensors ===")
t = torch.tensor([1., 2., 3.])
print(t, t.dtype, t.shape)

m = torch.zeros(3, 4)
r = torch.randn(2, 3)
print(m, r)

# Ops
print(t * 2)           # element-wise
print(t @ t)           # dot product
print(r.T)             # transpose
print(r.sum(dim=0))    # sum along rows

# NumPy ↔ PyTorch
arr = np.array([1., 2., 3.])
t_from_np = torch.from_numpy(arr)
back_to_np = t_from_np.numpy()
print(type(t_from_np), type(back_to_np))

# ─── 2. AUTOGRAD ─────────────────────────────────────────────────────────────
print("\n=== 2. Autograd ===")
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2*x + 1    # y = (x+1)^2
y.backward()             # compute dy/dx
print(f"x=3: y={y.item():.2f}, dy/dx={x.grad.item():.2f}")   # should be 2*(3+1)=8

# Disable gradient tracking for inference
with torch.no_grad():
    z = x * 2            # no gradient tracked

# ─── 3. BUILDING A NETWORK ───────────────────────────────────────────────────
print("\n=== 3. Building a Network with nn.Module ===")
class SimpleNet(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)

model = SimpleNet(10, 32, 1)
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# ─── 4. TRAINING LOOP ANATOMY ────────────────────────────────────────────────
print("\n=== 4. Training Loop (XOR as demo) ===")
X_xor = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y_xor = torch.tensor([0., 1., 1., 0.]).unsqueeze(1)

tiny = nn.Sequential(nn.Linear(2, 8), nn.Sigmoid(), nn.Linear(8, 1), nn.Sigmoid())
optimizer = optim.Adam(tiny.parameters(), lr=0.05)
criterion = nn.BCELoss()

for epoch in range(1, 2001):
    optimizer.zero_grad()           # clear previous gradients
    output = tiny(X_xor)            # forward pass
    loss   = criterion(output, y_xor)  # compute loss
    loss.backward()                 # backprop
    optimizer.step()                # update weights
    if epoch % 500 == 0:
        pred = (output > 0.5).float()
        acc  = (pred == y_xor).float().mean()
        print(f"Epoch {epoch}: loss={loss.item():.4f}, acc={acc.item():.4f}")

print("XOR predictions:", tiny(X_xor).detach().numpy().flatten().round(2))

# ─── 5. REAL DATA — DIGITS ───────────────────────────────────────────────────
print("\n=== 5. Digits (subset 2000 samples) ===")
# ~300 MB RAM, ~60s on CPU for 10 epochs

digits = load_digits()
idx = np.random.choice(len(digits.data), 2000, replace=False)
X, y = digits.data[idx].astype(np.float32), digits.target[idx]

sc = StandardScaler()
X = sc.fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_tr_t = torch.FloatTensor(X_tr); y_tr_t = torch.LongTensor(y_tr)
X_te_t = torch.FloatTensor(X_te); y_te_t = torch.LongTensor(y_te)

dataset  = TensorDataset(X_tr_t, y_tr_t)
loader   = DataLoader(dataset, batch_size=32, shuffle=True)

digit_net = nn.Sequential(
    nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 10)
)
optimizer = optim.Adam(digit_net.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

losses = []
for epoch in range(1, 11):   # 10 epochs — fast on CPU with 2000 samples
    digit_net.train()
    epoch_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        out  = digit_net(X_batch)
        loss = criterion(out, y_batch)
        loss.backward(); optimizer.step()
        epoch_loss += loss.item()
    losses.append(epoch_loss / len(loader))

    digit_net.eval()
    with torch.no_grad():
        preds = digit_net(X_te_t).argmax(dim=1)
        acc   = (preds == y_te_t).float().mean()
    print(f"Epoch {epoch:2d}: loss={losses[-1]:.4f}, test_acc={acc:.4f}")

# ─── 6. VISUALISE ────────────────────────────────────────────────────────────
plt.plot(losses, marker="o"); plt.title("Training Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.tight_layout(); plt.savefig("pytorch_digits.png", dpi=80); plt.close()
print("\nSaved pytorch_digits.png")
