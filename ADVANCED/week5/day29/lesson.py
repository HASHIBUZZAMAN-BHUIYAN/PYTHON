# Day 29: RNN Fundamentals
# ~150 MB RAM, ~20s on CPU
#
# Topics:
#   1. Sequence data intuition
#   2. Vanilla RNN from scratch (NumPy) — forward + BPTT
#   3. PyTorch nn.RNN
#   4. Vanishing gradient demo

import numpy as np
import torch
import torch.nn as nn
import time

print("=" * 60)
print("DAY 29: RNN FUNDAMENTALS")
print("=" * 60)

# ─────────────────────────────────────────────
# SECTION 1: SEQUENCE DATA INTUITION
# ─────────────────────────────────────────────
print("\n--- SECTION 1: Sequence Data Intuition ---")
print("""
Sequence data: order matters!
  - Text: "dog bites man" ≠ "man bites dog"
  - Time series: stock prices, sensor readings
  - Audio: speech waveforms

A standard MLP treats each input independently.
An RNN carries a "hidden state" (memory) from step to step.

Architecture:
  h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
  y_t = W_hy @ h_t + b_y

  h_t  : hidden state at time t  (shape: hidden_size)
  x_t  : input at time t         (shape: input_size)
  W_hh : recurrent weight matrix (hidden_size × hidden_size)
  W_xh : input weight matrix     (hidden_size × input_size)
""")

# ─────────────────────────────────────────────
# SECTION 2: VANILLA RNN FROM SCRATCH (NumPy)
# ─────────────────────────────────────────────
print("\n--- SECTION 2: Vanilla RNN from Scratch (NumPy) ---")

class VanillaRNN:
    """
    Minimal character-level RNN implemented purely in NumPy.
    Demonstrates forward pass and BPTT (backprop through time).
    """
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.hidden_size = hidden_size
        self.lr = lr
        # Xavier-style init
        scale = 0.01
        self.W_xh = np.random.randn(hidden_size, input_size)  * scale
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale
        self.W_hy = np.random.randn(output_size, hidden_size) * scale
        self.b_h  = np.zeros((hidden_size, 1))
        self.b_y  = np.zeros((output_size, 1))

    def forward(self, inputs, h_prev):
        """
        inputs : list of one-hot vectors (each shape: [input_size, 1])
        Returns: list of outputs, list of hidden states, list of pre-activations
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)

        for t, x in enumerate(inputs):
            xs[t] = x
            hs[t] = np.tanh(self.W_xh @ x + self.W_hh @ hs[t-1] + self.b_h)
            ys[t] = self.W_hy @ hs[t] + self.b_y
            e = np.exp(ys[t] - np.max(ys[t]))
            ps[t] = e / e.sum()

        return xs, hs, ys, ps

    def bptt(self, inputs, targets, h_prev):
        """Backpropagation Through Time."""
        xs, hs, ys, ps = self.forward(inputs, h_prev)
        T = len(inputs)

        # Initialize gradients
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h  = np.zeros_like(self.b_h)
        db_y  = np.zeros_like(self.b_y)
        dh_next = np.zeros((self.hidden_size, 1))

        loss = 0.0
        grad_norms = []  # track gradient magnitude per timestep

        for t in reversed(range(T)):
            loss -= np.log(ps[t][targets[t], 0] + 1e-9)
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            dW_hy += dy @ hs[t].T
            db_y  += dy
            # hidden gradient
            dh = self.W_hy.T @ dy + dh_next
            dh_raw = (1 - hs[t] ** 2) * dh   # tanh' = 1 - tanh²
            db_h  += dh_raw
            dW_xh += dh_raw @ xs[t].T
            dW_hh += dh_raw @ hs[t-1].T
            dh_next = self.W_hh.T @ dh_raw
            grad_norms.append(float(np.linalg.norm(dh_next)))

        for param in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(param, -5, 5, out=param)

        self.W_xh -= self.lr * dW_xh
        self.W_hh -= self.lr * dW_hh
        self.W_hy -= self.lr * dW_hy
        self.b_h  -= self.lr * db_h
        self.b_y  -= self.lr * db_y

        return loss, hs[T-1], list(reversed(grad_norms))


# Tiny demo: learn to predict next character in "abcabc..."
print("\nDemo: NumPy RNN learning 'abcabc...' pattern")
chars  = ['a', 'b', 'c']
ch2ix  = {c: i for i, c in enumerate(chars)}
ix2ch  = {i: c for i, c in enumerate(chars)}
vocab  = len(chars)

def one_hot(idx, size):
    v = np.zeros((size, 1))
    v[idx] = 1.0
    return v

rnn_np = VanillaRNN(input_size=vocab, hidden_size=8, output_size=vocab, lr=0.05)
text   = "abcabcabcabc"
seq    = [ch2ix[c] for c in text]

h = np.zeros((8, 1))
for epoch in range(100):
    inputs  = [one_hot(seq[i],   vocab) for i in range(len(seq)-1)]
    targets = [seq[i+1]                 for i in range(len(seq)-1)]
    loss, h, _ = rnn_np.bptt(inputs, targets, np.zeros((8,1)))
    if epoch % 25 == 0:
        print(f"  Epoch {epoch:3d} | Loss: {loss:.4f}")

print("  NumPy RNN training complete.")

# ─────────────────────────────────────────────
# SECTION 3: PyTorch nn.RNN
# ─────────────────────────────────────────────
print("\n--- SECTION 3: PyTorch nn.RNN ---")

print("""
PyTorch nn.RNN replaces all the manual weight math.

Usage:
  rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
  output, h_n = rnn(x, h_0)

  x      : [batch, seq_len, input_size]  (if batch_first=True)
  h_0    : [num_layers, batch, hidden_size]
  output : [batch, seq_len, hidden_size]
  h_n    : [num_layers, batch, hidden_size]  — final hidden state
""")

torch.manual_seed(42)
input_size  = 4
hidden_size = 8
seq_len     = 6
batch_size  = 2

rnn_pt = nn.RNN(input_size, hidden_size, batch_first=True)
x      = torch.randn(batch_size, seq_len, input_size)
h0     = torch.zeros(1, batch_size, hidden_size)

output, h_n = rnn_pt(x, h0)
print(f"  Input shape:  {x.shape}")
print(f"  Output shape: {output.shape}  (batch, seq_len, hidden)")
print(f"  h_n shape:    {h_n.shape}     (num_layers, batch, hidden)")
print(f"  Parameters:   {sum(p.numel() for p in rnn_pt.parameters())}")

# ─────────────────────────────────────────────
# SECTION 4: VANISHING GRADIENT DEMO
# ─────────────────────────────────────────────
print("\n--- SECTION 4: Vanishing Gradient Demo ---")

print("""
The vanishing gradient problem:
  - Gradients are multiplied by W_hh^T at each BPTT step
  - If |eigenvalues of W_hh| < 1 → gradients shrink exponentially
  - Early timesteps receive nearly zero gradient → RNN can't learn long deps

We'll measure gradient norms across timesteps for a 30-step sequence.
""")

class GradientTracker(nn.Module):
    def __init__(self, hidden_size, seq_len):
        super().__init__()
        self.rnn  = nn.RNN(1, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
        self.seq_len = seq_len

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

seq_len     = 30
hidden_size = 16
model       = GradientTracker(hidden_size, seq_len)
optimizer   = torch.optim.Adam(model.parameters(), lr=0.01)

# synthetic task: remember first element, predict it at end
x_data = torch.randn(64, seq_len, 1)
y_data = x_data[:, 0, :]   # label = first element

grad_norms_per_step = []

for step in range(50):
    pred = model(x_data)
    loss = nn.MSELoss()(pred, y_data)
    optimizer.zero_grad()
    loss.backward()

    # collect gradient norms for each timestep's hidden state
    # (approximated via output gradients at each time position)
    with torch.no_grad():
        out, _ = model.rnn(x_data)
        out.retain_grad()

    if step == 49:
        out2, _ = model.rnn(x_data)
        out2.retain_grad()
        pred2 = model.fc(out2[:, -1, :])
        loss2 = nn.MSELoss()(pred2, y_data)
        loss2.backward()
        if out2.grad is not None:
            norms = out2.grad.abs().mean(dim=[0, 2]).detach().numpy()
            grad_norms_per_step = norms

    optimizer.step()

if len(grad_norms_per_step) > 0:
    print("  Gradient norms across timesteps (t=0 is earliest):")
    for t in [0, 5, 10, 15, 20, 25, 29]:
        bar = "█" * max(1, int(grad_norms_per_step[t] * 200))
        print(f"  t={t:2d}: {grad_norms_per_step[t]:.6f}  {bar[:40]}")
    print(f"\n  Ratio (t=0 / t=29): {grad_norms_per_step[0]/max(grad_norms_per_step[29], 1e-9):.4f}")
    print("  → Early timesteps have near-zero gradients = vanishing gradient!")

print("""
Solution: LSTM and GRU (Day 30) use gating mechanisms to control
gradient flow, allowing learning over much longer sequences.
""")

print("\n" + "=" * 60)
print("Day 29 lesson complete!")
print("Key takeaways:")
print("  1. RNNs maintain hidden state across sequence steps")
print("  2. BPTT multiplies gradients across every timestep")
print("  3. Gradients vanish exponentially for long sequences")
print("  4. PyTorch nn.RNN handles all weight math cleanly")
print("=" * 60)
