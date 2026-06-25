# Advanced Day 08 — Neural Networks from Scratch (Pure NumPy)
# ~50 MB RAM, <10s on CPU

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ─── 1. ACTIVATION FUNCTIONS ────────────────────────────────────────────────
def sigmoid(z):    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def sigmoid_d(z):  s = sigmoid(z); return s * (1 - s)
def relu(z):       return np.maximum(0, z)
def relu_d(z):     return (z > 0).astype(float)
def tanh(z):       return np.tanh(z)
def tanh_d(z):     return 1 - np.tanh(z)**2

# ─── 2. LOSS FUNCTIONS ───────────────────────────────────────────────────────
def mse(y, y_hat):     return np.mean((y - y_hat)**2)
def mse_d(y, y_hat):   return 2 * (y_hat - y) / y.size
def bce(y, y_hat):
    eps = 1e-8
    return -np.mean(y * np.log(y_hat+eps) + (1-y)*np.log(1-y_hat+eps))

# ─── 3. TWO-LAYER NEURAL NETWORK CLASS ──────────────────────────────────────
class TwoLayerNet:
    """Input → Hidden (sigmoid) → Output (sigmoid). Binary classification."""

    def __init__(self, n_input, n_hidden, n_output=1, lr=0.1):
        # He/Glorot-like initialization
        self.W1 = np.random.randn(n_input, n_hidden) * np.sqrt(2.0 / n_input)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_output) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros((1, n_output))
        self.lr = lr
        self.loss_history = []

    def forward(self, X):
        self.X   = X
        self.Z1  = X @ self.W1 + self.b1         # pre-activation hidden
        self.A1  = sigmoid(self.Z1)               # hidden activations
        self.Z2  = self.A1 @ self.W2 + self.b2   # pre-activation output
        self.A2  = sigmoid(self.Z2)               # output (predictions)
        return self.A2

    def backward(self, y):
        m = y.shape[0]
        # Output layer gradients
        dZ2 = self.A2 - y.reshape(-1, 1)          # BCE derivative shortcut
        dW2 = self.A1.T @ dZ2 / m
        db2 = dZ2.mean(axis=0, keepdims=True)

        # Hidden layer gradients
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * sigmoid_d(self.Z1)
        dW1 = self.X.T @ dZ1 / m
        db1 = dZ1.mean(axis=0, keepdims=True)

        # Update weights
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=1000, print_every=200):
        for epoch in range(1, epochs+1):
            y_hat = self.forward(X)
            loss  = bce(y, y_hat.flatten())
            self.loss_history.append(loss)
            self.backward(y)
            if epoch % print_every == 0:
                acc = ((y_hat.flatten() > 0.5) == y).mean()
                print(f"Epoch {epoch:4d}: loss={loss:.4f}, acc={acc:.4f}")

    def predict(self, X):
        return (self.forward(X).flatten() > 0.5).astype(int)


# ─── 4. TOY DATASET — XOR ────────────────────────────────────────────────────
print("=== XOR Problem (not linearly separable) ===")
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([0, 1, 1, 0], dtype=float)

net = TwoLayerNet(n_input=2, n_hidden=4, lr=0.5)
net.train(X_xor, y_xor, epochs=5000, print_every=1000)
print("Predictions:", net.predict(X_xor), "True:", y_xor.astype(int))

# ─── 5. CIRCULAR DATASET ────────────────────────────────────────────────────
from sklearn.datasets import make_circles
X_c, y_c = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)
X_c = (X_c - X_c.mean(axis=0)) / X_c.std(axis=0)  # normalize

net2 = TwoLayerNet(n_input=2, n_hidden=8, lr=0.3)
print("\n=== Circles Dataset ===")
net2.train(X_c, y_c, epochs=2000, print_every=500)
print("Final accuracy:", (net2.predict(X_c) == y_c).mean())

# ─── 6. VISUALISE ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(net.loss_history, color="steelblue")
axes[0].set_title("XOR Training Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")

axes[1].plot(net2.loss_history, color="tomato")
axes[1].set_title("Circles Training Loss")

# Decision boundary for circles
h = 0.05
x_min, x_max = X_c[:,0].min()-0.5, X_c[:,0].max()+0.5
y_min, y_max = X_c[:,1].min()-0.5, X_c[:,1].max()+0.5
xx, yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max,h))
Z = net2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[2].contourf(xx,yy,Z,alpha=0.3,cmap="RdBu")
axes[2].scatter(X_c[:,0],X_c[:,1],c=y_c,cmap="RdBu",edgecolors="k",s=20)
axes[2].set_title("Decision Boundary (Circles)")

plt.tight_layout(); plt.savefig("nn_scratch.png", dpi=80)
print("\nSaved nn_scratch.png")
