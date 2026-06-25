# Day 29: RNN Fundamentals — Solutions
# Full working solutions for all 5 exercises.

import numpy as np
import torch
import torch.nn as nn

print("Day 29 Solutions\n")

# ─────────────────────────────────────────────
# Solution 1: RNN output shapes
# ─────────────────────────────────────────────
print("Solution 1: RNN output shapes")

rnn = nn.RNN(input_size=5, hidden_size=20, batch_first=True)
x   = torch.randn(4, 8, 5)          # [batch=4, seq_len=8, input_size=5]
h0  = torch.zeros(1, 4, 20)         # [num_layers=1, batch=4, hidden=20]

output, h_n = rnn(x, h0)
print(f"  output shape: {output.shape}")   # [4, 8, 20]
print(f"  h_n shape:    {h_n.shape}")      # [1, 4, 20]

# ─────────────────────────────────────────────
# Solution 2: Manual hidden state update
# ─────────────────────────────────────────────
print("\nSolution 2: Manual hidden state update")

np.random.seed(0)
input_size  = 3
hidden_size = 4

W_xh  = np.random.randn(hidden_size, input_size)  * 0.1
W_hh  = np.random.randn(hidden_size, hidden_size) * 0.1
b     = np.zeros(hidden_size)

x_t    = np.random.randn(input_size)
h_prev = np.zeros(hidden_size)

# RNN step: h_t = tanh(W_xh @ x_t + W_hh @ h_prev + b)
h_t = np.tanh(W_xh @ x_t + W_hh @ h_prev + b)
print(f"  h_t shape:  {h_t.shape}")
print(f"  h_t values: {np.round(h_t, 4)}")

# ─────────────────────────────────────────────
# Solution 3: Sequence-to-scalar predictor
# ─────────────────────────────────────────────
print("\nSolution 3: Sequence-to-scalar predictor")

torch.manual_seed(42)

class SumPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=8, batch_first=True)
        self.fc  = nn.Linear(8, 1)

    def forward(self, x):
        # x: [batch, seq_len, 1]
        _, h_n = self.rnn(x)           # h_n: [1, batch, 8]
        return self.fc(h_n.squeeze(0)) # [batch, 1]

# Synthetic data: 50 sequences of length 10
n_samples = 50
seq_len   = 10
X = torch.randn(n_samples, seq_len, 1)
y = X.sum(dim=1)                       # [50, 1] — sum of sequence

model    = SumPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(30):
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"  Epoch {epoch:2d} | Loss: {loss.item():.4f}")

print(f"  Final loss: {loss.item():.4f}")

# ─────────────────────────────────────────────
# Solution 4: Gradient clipping
# ─────────────────────────────────────────────
print("\nSolution 4: Gradient clipping")

torch.manual_seed(42)
model4    = SumPredictor()
optimizer4 = torch.optim.Adam(model4.parameters(), lr=0.01)

# one forward + backward pass
pred  = model4(X)
loss  = nn.MSELoss()(pred, y)
optimizer4.zero_grad()
loss.backward()

# compute gradient norm BEFORE clipping
total_norm_before = sum(
    p.grad.norm() ** 2 for p in model4.parameters() if p.grad is not None
) ** 0.5
print(f"  Gradient norm BEFORE clipping: {total_norm_before.item():.4f}")

# clip gradients
torch.nn.utils.clip_grad_norm_(model4.parameters(), max_norm=1.0)

# compute gradient norm AFTER clipping
total_norm_after = sum(
    p.grad.norm() ** 2 for p in model4.parameters() if p.grad is not None
) ** 0.5
print(f"  Gradient norm AFTER  clipping: {total_norm_after.item():.4f}")
print(f"  (should be ≤ 1.0 after clipping)")

# ─────────────────────────────────────────────
# Solution 5: Multi-layer RNN
# ─────────────────────────────────────────────
print("\nSolution 5: Multi-layer RNN")

rnn2 = nn.RNN(input_size=4, hidden_size=16, num_layers=2, batch_first=True)
x2   = torch.randn(2, 6, 4)   # [batch=2, seq=6, features=4]
out2, h_n2 = rnn2(x2)

print(f"  h_n shape: {h_n2.shape}")
# h_n shape: [2, 2, 16]
# Dimension 0: num_layers=2  (one hidden state per layer)
# Dimension 1: batch_size=2
# Dimension 2: hidden_size=16
print("  Explanation:")
print("    h_n[0] = final hidden state from layer 1 (bottom RNN)")
print("    h_n[1] = final hidden state from layer 2 (top RNN)")
print("    Each has shape [batch=2, hidden=16]")

print("\nAll solutions complete!")
