# Day 30: LSTM & GRU — Solutions

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

print("Day 30 Solutions\n")

# ─────────────────────────────────────────────
# Solution 1: LSTM vs GRU Parameter Count
# ─────────────────────────────────────────────
print("Solution 1: LSTM vs GRU parameter count")

input_size  = 10
hidden_size = 32

lstm = nn.LSTM(input_size, hidden_size)
gru  = nn.GRU(input_size, hidden_size)

lstm_params = sum(p.numel() for p in lstm.parameters())
gru_params  = sum(p.numel() for p in gru.parameters())

print(f"  LSTM params: {lstm_params}")
print(f"  GRU  params: {gru_params}")

# Manual verification:
# PyTorch LSTM has weight_ih, weight_hh, bias_ih, bias_hh per layer
# weight_ih: [4*hidden, input]  = 4*32*10 = 1280
# weight_hh: [4*hidden, hidden] = 4*32*32 = 4096
# bias_ih:   [4*hidden]         = 128
# bias_hh:   [4*hidden]         = 128
# Total LSTM = 1280+4096+128+128 = 5632
lstm_manual = 4 * hidden_size * input_size + 4 * hidden_size * hidden_size + 2 * 4 * hidden_size
gru_manual  = 3 * hidden_size * input_size + 3 * hidden_size * hidden_size + 2 * 3 * hidden_size
print(f"  LSTM manual check: {lstm_manual}  (matches: {lstm_manual == lstm_params})")
print(f"  GRU  manual check: {gru_manual}   (matches: {gru_manual  == gru_params})")

# ─────────────────────────────────────────────
# Solution 2: Bidirectional LSTM
# ─────────────────────────────────────────────
print("\nSolution 2: Bidirectional LSTM")

bilstm = nn.LSTM(input_size=6, hidden_size=12, batch_first=True, bidirectional=True)
x      = torch.randn(2, 8, 6)
out, (h_n, c_n) = bilstm(x)

print(f"  Output shape: {out.shape}")
# output has size 24 = 12 * 2 because:
# forward  LSTM produces hidden_size=12 features per timestep
# backward LSTM produces hidden_size=12 features per timestep
# they are CONCATENATED at each timestep → 24 total
print("  Why 24? forward(12) + backward(12) = 24 (concat at each timestep)")
print(f"  h_n shape: {h_n.shape}")
# [2, 2, 12] = [num_layers * num_directions, batch, hidden]

# ─────────────────────────────────────────────
# Solution 3: LSTM Sequence Classifier
# ─────────────────────────────────────────────
print("\nSolution 3: LSTM sequence classifier")

torch.manual_seed(42)
n_samples = 80
seq_len   = 15

X = torch.randn(n_samples, seq_len, 1)
y = (X.mean(dim=1) > 0).long().squeeze()   # [80]

class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.fc   = nn.Linear(16, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # last timestep

model     = LSTMClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    logits = model(X)
    loss   = criterion(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    preds    = model(X).argmax(dim=1)
    accuracy = (preds == y).float().mean().item()
print(f"  Final accuracy: {accuracy:.2%}")

# ─────────────────────────────────────────────
# Solution 4: GRU with Dropout
# ─────────────────────────────────────────────
print("\nSolution 4: GRU with dropout")

gru_drop = nn.GRU(input_size=8, hidden_size=24, num_layers=2, dropout=0.3, batch_first=True)
x4       = torch.randn(3, 12, 8)
out4, _  = gru_drop(x4)

print(f"  GRU (2 layers, dropout=0.3) params: {sum(p.numel() for p in gru_drop.parameters())}")
print(f"  Output shape: {out4.shape}")
print("  Shape is same as without dropout — dropout doesn't change tensor dimensions")
print("  Dropout only randomly zeros activations during training (not eval)")

# ─────────────────────────────────────────────
# Solution 5: Packed Sequences
# ─────────────────────────────────────────────
print("\nSolution 5: Packed sequences")

# 3 sequences, padded to length 6. Lengths must be sorted descending.
lengths = [6, 4, 3]
max_len = max(lengths)
feat    = 4

# Build padded tensor [3, 6, 4] — sorted by length (descending) already
x5 = torch.zeros(3, max_len, feat)
for i, L in enumerate(lengths):
    x5[i, :L, :] = torch.randn(L, feat)

print(f"  Padded input shape: {x5.shape}")

# Pack
lengths_tensor = torch.tensor(lengths)
packed = pack_padded_sequence(x5, lengths_tensor, batch_first=True, enforce_sorted=True)

# Run LSTM
lstm5   = nn.LSTM(input_size=feat, hidden_size=8, batch_first=True)
packed_out, _ = lstm5(packed)

# Unpack
output, out_lengths = pad_packed_sequence(packed_out, batch_first=True)
print(f"  Unpacked output shape: {output.shape}")
print(f"  Recovered lengths: {out_lengths.tolist()}")
print("  Key benefit: RNN only processes real tokens, not padding zeros!")

print("\nAll solutions complete!")
