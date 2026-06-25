# Advanced Day 08 Mini-Project — Handwritten Digit Classifier (Pure NumPy, 2 classes)
# ~80 MB RAM, <20s on CPU

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# Use digits 0 vs 1 (binary classification, keeps it simple)
digits = load_digits()
mask = digits.target < 2
X, y = digits.data[mask], digits.target[mask]
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ─── Three-layer network ─────────────────────────────────────────────────────
def sigmoid(z):   return 1 / (1 + np.exp(-np.clip(z,-500,500)))
def relu(z):      return np.maximum(0, z)
def relu_d(z):    return (z > 0).astype(float)

class ThreeLayerNet:
    def __init__(self, sizes, lr=0.05):
        n0, n1, n2, n3 = sizes
        self.W = [np.random.randn(n0,n1)*np.sqrt(2/n0),
                  np.random.randn(n1,n2)*np.sqrt(2/n1),
                  np.random.randn(n2,n3)*np.sqrt(2/n2)]
        self.b = [np.zeros((1,s)) for s in [n1,n2,n3]]
        self.lr = lr; self.losses = []

    def forward(self, X):
        self.cache = [X]
        A = X
        for i in range(2):
            Z = A @ self.W[i] + self.b[i]; A = relu(Z)
            self.cache.append(Z); self.cache.append(A)
        Z_out = A @ self.W[2] + self.b[2]
        A_out = sigmoid(Z_out)
        self.cache.append(Z_out); self.cache.append(A_out)
        return A_out

    def backward(self, y):
        m = y.shape[0]
        A_out = self.cache[-1]
        dZ = (A_out - y.reshape(-1,1)) / m
        grads_W, grads_b = [], []
        A_prev = self.cache[-3]  # A2
        grads_W.insert(0, A_prev.T @ dZ)
        grads_b.insert(0, dZ.sum(axis=0, keepdims=True))
        for layer in [1, 0]:
            dA = dZ @ self.W[layer+1].T
            Z_prev = self.cache[1 + layer*2]
            dZ = dA * relu_d(Z_prev)
            A_prev = self.cache[layer*2]
            grads_W.insert(0, A_prev.T @ dZ)
            grads_b.insert(0, dZ.sum(axis=0, keepdims=True))
        for i in range(3):
            self.W[i] -= self.lr * grads_W[i]
            self.b[i] -= self.lr * grads_b[i]

    def train(self, X, y, epochs=500):
        for ep in range(1, epochs+1):
            y_hat = self.forward(X)
            eps = 1e-8
            loss = -np.mean(y*np.log(y_hat+eps)+(1-y)*np.log(1-y_hat+eps))
            self.losses.append(loss)
            self.backward(y)
            if ep % 100 == 0:
                acc = ((y_hat.flatten()>0.5) == y).mean()
                print(f"  Epoch {ep:4d}: loss={loss:.4f}  acc={acc:.4f}")

    def predict(self, X):
        return (self.forward(X).flatten() > 0.5).astype(int)

print("Training 64→32→16→1 network on digits 0 vs 1 ...")
net = ThreeLayerNet([64, 32, 16, 1], lr=0.05)
net.train(X_train, y_train, epochs=500)

test_acc = (net.predict(X_test) == y_test).mean()
print(f"\nTest accuracy: {test_acc:.4f}")

# ─── Visualise ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(net.losses, color="steelblue")
axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch")

# Show some predictions
incorrect_idx = np.where(net.predict(X_test) != y_test)[0]
fig2, axs = plt.subplots(2, 5, figsize=(10, 4))
fig2.suptitle(f"Sample predictions (test acc={test_acc:.3f})")
for i, ax in enumerate(axs.flat):
    idx = i if i < len(X_test) else 0
    ax.imshow(X_test[idx].reshape(8,8), cmap="gray")
    pred = net.predict(X_test[idx:idx+1])[0]
    true = y_test[idx]
    color = "green" if pred==true else "red"
    ax.set_title(f"P:{pred} T:{true}", color=color, fontsize=8)
    ax.axis("off")

plt.tight_layout(); fig.savefig("digits_nn.png", dpi=80); fig2.savefig("digits_preds.png", dpi=80)
print("Saved digits_nn.png, digits_preds.png")
plt.show()
