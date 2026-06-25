# Advanced Day 10 Mini-Project — Keras vs PyTorch Side-by-Side Benchmark
# Trains the same architecture on the same data in both frameworks.
# Compares accuracy, training speed, and code complexity.
# ~400 MB RAM, ~5 min on CPU

import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as klayers
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(42); tf.random.set_seed(42); torch.manual_seed(42)

EPOCHS = 20; BATCH = 32; LR = 1e-3
digits = load_digits()
sc = StandardScaler()
X = sc.fit_transform(digits.data).astype(np.float32); y = digits.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("="*55); print("KERAS vs PYTORCH BENCHMARK"); print("="*55)

# ─── KERAS ───────────────────────────────────────────────────────────────────
k_model = keras.Sequential([
    klayers.Input(64), klayers.Dense(256,"relu"), klayers.BatchNormalization(),
    klayers.Dropout(0.3), klayers.Dense(128,"relu"), klayers.Dense(10,"softmax")
])
k_model.compile(keras.optimizers.Adam(LR), "sparse_categorical_crossentropy", ["accuracy"])
t0 = time.time()
k_hist = k_model.fit(Xtr, ytr, EPOCHS, BATCH, validation_data=(Xte,yte), verbose=0)
k_time = time.time() - t0
k_acc = k_model.evaluate(Xte, yte, verbose=0)[1]
print(f"Keras   — test_acc={k_acc:.4f}  time={k_time:.1f}s")

# ─── PYTORCH ─────────────────────────────────────────────────────────────────
p_model = nn.Sequential(
    nn.Linear(64,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256,128), nn.ReLU(), nn.Linear(128,10)
)
p_opt = optim.Adam(p_model.parameters(), LR)
p_crit = nn.CrossEntropyLoss()
loader = DataLoader(TensorDataset(torch.FloatTensor(Xtr), torch.LongTensor(ytr)), BATCH, shuffle=True)
p_te_X, p_te_y = torch.FloatTensor(Xte), torch.LongTensor(yte)
p_tr_hist, p_val_hist = [], []
t0 = time.time()
for ep in range(EPOCHS):
    p_model.train(); tl=0
    for xb,yb in loader:
        p_opt.zero_grad(); out=p_model(xb); l=p_crit(out,yb); l.backward(); p_opt.step(); tl+=l.item()
    p_tr_hist.append(tl/len(loader))
    p_model.eval()
    with torch.no_grad(): p_val_hist.append((p_model(p_te_X).argmax(1)==p_te_y).float().mean().item())
p_time = time.time() - t0
p_acc = p_val_hist[-1]
print(f"PyTorch — test_acc={p_acc:.4f}  time={p_time:.1f}s")

# ─── Visualise ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Keras vs PyTorch — same architecture, same data")

axes[0].plot(k_hist.history["val_accuracy"], label=f"Keras ({k_acc:.3f})", linewidth=2)
axes[0].plot(p_val_hist, label=f"PyTorch ({p_acc:.3f})", linewidth=2, linestyle="--")
axes[0].set_title("Validation Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend()

axes[1].plot(k_hist.history["loss"], label="Keras train")
axes[1].plot(k_hist.history["val_loss"], label="Keras val", linestyle="--")
axes[1].plot(p_tr_hist, label="PyTorch train")
axes[1].set_title("Training Loss"); axes[1].set_xlabel("Epoch"); axes[1].legend()

plt.tight_layout(); plt.savefig("framework_comparison.png", dpi=80)
print("\nSaved framework_comparison.png")
plt.show()
