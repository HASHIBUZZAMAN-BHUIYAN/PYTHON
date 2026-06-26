# Day 30: LSTM & GRU
# ~180 MB RAM, ~25s on CPU
#
# Topics:
#   1. Why vanilla RNN fails on long sequences
#   2. LSTM cell gates — intuitive explanation
#   3. GRU as simpler alternative
#   4. PyTorch nn.LSTM and nn.GRU
#   5. Compare RNN vs LSTM on long-sequence task

import torch
import torch.nn as nn
import numpy as np
import time

print("=" * 60)
print("DAY 30: LSTM & GRU")
print("=" * 60)

# ─────────────────────────────────────────────
# SECTION 1: WHY VANILLA RNN FAILS
# ─────────────────────────────────────────────
print("\n--- SECTION 1: Why Vanilla RNN Fails on Long Sequences ---")
print("""
Recall from Day 29: the vanishing gradient problem.

In BPTT, the gradient at timestep t is:
  ∂L/∂h_0 = ∂L/∂h_T · (∏_{t=1}^{T} ∂h_t/∂h_{t-1})

Each ∂h_t/∂h_{t-1} = diag(1 - h_t²) · W_hh

If the spectral radius of W_hh < 1:
  → gradient shrinks exponentially → vanishes
  → early inputs are "forgotten"

If spectral radius > 1:
  → gradient explodes → training instability

This means vanilla RNNs can practically only remember ~10-20 steps.
For tasks requiring 50-100+ steps of memory → LSTM or GRU.
""")

# ─────────────────────────────────────────────
# SECTION 2: LSTM GATES
# ─────────────────────────────────────────────
print("\n--- SECTION 2: LSTM Cell Gates ---")
print("""
LSTM (Long Short-Term Memory) — Hochreiter & Schmidhuber, 1997

Key insight: maintain a CELL STATE c_t that flows almost unchanged
through time (the "memory highway"). Gates control what to remember,
forget, and output.

Three gates (each produces values in [0,1] via sigmoid):

  FORGET GATE  f_t: "How much of the old memory to keep?"
    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
    → 0 = completely forget, 1 = completely keep

  INPUT GATE   i_t: "How much of the new candidate to add?"
    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
    g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)   ← new candidate values

  OUTPUT GATE  o_t: "How much of the cell state to expose as h_t?"
    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

Cell state update (the magic!):
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
           ↑ forget old   ↑ add new
    h_t = o_t ⊙ tanh(c_t)

Why does this help with gradients?
  ∂c_t/∂c_{t-1} = f_t  (just a multiplication — no matrix!)
  → gradient flows back through c unchanged if f_t ≈ 1
  → LSTM can maintain gradients over 100s of steps
""")

# Manual LSTM step in NumPy to make it concrete
print("Manual LSTM step (NumPy demo):")
np.random.seed(42)
input_size  = 4
hidden_size = 8
batch_size  = 1

# Weights for all 4 gates combined: shape [4*hidden, hidden+input]
combined_size = hidden_size + input_size
W = np.random.randn(4 * hidden_size, combined_size) * 0.1
b = np.zeros(4 * hidden_size)

def sigmoid(x): return 1 / (1 + np.exp(-x))

def lstm_step(x, h_prev, c_prev, W, b):
    combined = np.concatenate([h_prev, x])
    gates    = W @ combined + b   # [4*hidden]
    h = hidden_size
    f = sigmoid(gates[0*h : 1*h])   # forget
    i = sigmoid(gates[1*h : 2*h])   # input
    g = np.tanh(gates[2*h : 3*h])   # candidate
    o = sigmoid(gates[3*h : 4*h])   # output
    c_new = f * c_prev + i * g
    h_new = o * np.tanh(c_new)
    return h_new, c_new

x      = np.random.randn(input_size)
h_prev = np.zeros(hidden_size)
c_prev = np.zeros(hidden_size)
h_new, c_new = lstm_step(x, h_prev, c_prev, W, b)
print(f"  h_new range: [{h_new.min():.3f}, {h_new.max():.3f}]")
print(f"  c_new range: [{c_new.min():.3f}, {c_new.max():.3f}]")

# ─────────────────────────────────────────────
# SECTION 3: GRU
# ─────────────────────────────────────────────
print("\n--- SECTION 3: GRU (Gated Recurrent Unit) ---")
print("""
GRU — Cho et al., 2014

Simpler than LSTM: 2 gates, no separate cell state.
Often matches LSTM performance with fewer parameters.

  RESET GATE  r_t = σ(W_r · [h_{t-1}, x_t])
    → How much past hidden state to use when computing candidate

  UPDATE GATE z_t = σ(W_z · [h_{t-1}, x_t])
    → How much to update the hidden state (like combined forget+input)

  CANDIDATE   h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
  NEW HIDDEN  h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
               ↑ keep old            ↑ add new

Comparison:
  LSTM: 4 gates, cell state + hidden state → more params
  GRU:  2 gates, only hidden state        → fewer params

Rule of thumb:
  - Long sequences, complex tasks → LSTM
  - Moderate sequences, limited data/compute → GRU
  - Both vastly outperform vanilla RNN on long-range deps
""")

# ─────────────────────────────────────────────
# SECTION 4: PyTorch nn.LSTM and nn.GRU
# ─────────────────────────────────────────────
print("\n--- SECTION 4: PyTorch nn.LSTM and nn.GRU ---")

torch.manual_seed(42)
batch_size  = 3
seq_len     = 10
input_size  = 5
hidden_size = 16

x = torch.randn(batch_size, seq_len, input_size)

# nn.LSTM
lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
h0   = torch.zeros(1, batch_size, hidden_size)
c0   = torch.zeros(1, batch_size, hidden_size)
lstm_out, (h_n, c_n) = lstm(x, (h0, c0))

print("nn.LSTM:")
print(f"  output: {lstm_out.shape}  [batch, seq_len, hidden]")
print(f"  h_n:    {h_n.shape}   [num_layers, batch, hidden]")
print(f"  c_n:    {c_n.shape}   [num_layers, batch, hidden]")
print(f"  params: {sum(p.numel() for p in lstm.parameters())}")

# nn.GRU
gru = nn.GRU(input_size, hidden_size, batch_first=True)
gru_out, h_n_gru = gru(x)
print("\nnn.GRU:")
print(f"  output: {gru_out.shape}  [batch, seq_len, hidden]")
print(f"  h_n:    {h_n_gru.shape}   [num_layers, batch, hidden]")
print(f"  params: {sum(p.numel() for p in gru.parameters())}")
print(f"  (GRU has ~3/4 the params of LSTM with same hidden size)")

# ─────────────────────────────────────────────
# SECTION 5: COMPARE RNN vs LSTM on LONG TASK
# ─────────────────────────────────────────────
print("\n--- SECTION 5: RNN vs LSTM on Long-Sequence Task ---")
print("Task: remember first element of sequence, predict it at the end")
print("Sequence length: 40  — requires long-range memory\n")

torch.manual_seed(42)
SEQ   = 40
N     = 500
HDIM  = 32
EPOCH = 30

X_data = torch.randn(N, SEQ, 1)
y_data = X_data[:, 0, :]   # label = first element

split  = 400
X_tr, y_tr = X_data[:split], y_data[:split]
X_te, y_te = X_data[split:], y_data[split:]

class MemoryModel(nn.Module):
    def __init__(self, cell_type="rnn", hidden_size=32):
        super().__init__()
        if cell_type == "rnn":
            self.rnn = nn.RNN(1, hidden_size, batch_first=True)
        elif cell_type == "lstm":
            self.rnn = nn.LSTM(1, hidden_size, batch_first=True)
        elif cell_type == "gru":
            self.rnn = nn.GRU(1, hidden_size, batch_first=True)
        self.fc       = nn.Linear(hidden_size, 1)
        self.cell_type = cell_type

    def forward(self, x):
        out = self.rnn(x)
        if self.cell_type == "lstm":
            h_n = out[1][0]   # LSTM returns (h_n, c_n); take h_n
        else:
            h_n = out[1]
        return self.fc(h_n.squeeze(0))

results = {}
for cell_type in ["rnn", "gru", "lstm"]:
    torch.manual_seed(42)
    m   = MemoryModel(cell_type, HDIM)
    opt = torch.optim.Adam(m.parameters(), lr=0.005)
    crit = nn.MSELoss()
    t0  = time.time()

    for ep in range(EPOCH):
        m.train()
        pred = m(X_tr)
        loss = crit(pred, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    m.eval()
    with torch.no_grad():
        test_loss = crit(m(X_te), y_te).item()
    elapsed = time.time() - t0
    results[cell_type] = (test_loss, elapsed)
    print(f"  {cell_type.upper():4s} | Test MSE: {test_loss:.4f} | Time: {elapsed:.1f}s")

print("\n  Lower MSE = better long-range memory.")
print("  LSTM/GRU should outperform vanilla RNN on this task.")

print("\n" + "=" * 60)
print("Day 30 lesson complete!")
print("Key takeaways:")
print("  1. LSTM cell state provides a 'gradient highway' → no vanishing")
print("  2. Gates (forget/input/output) learn what to remember/forget")
print("  3. GRU is simpler than LSTM, often comparable performance")
print("  4. Both far outperform RNN on long-sequence tasks")
print("=" * 60)
