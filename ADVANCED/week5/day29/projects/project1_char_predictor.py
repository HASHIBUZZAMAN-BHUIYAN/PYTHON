"""
Day 29 Project 1: Character-Level Next-Char Predictor
======================================================
Train a small RNN to predict the next character in a poem.
Vocabulary: a-z + space (27 characters).
Architecture: nn.RNN, hidden_size=32
Training: 20 epochs on a hardcoded ~200 char string.
After training: generate sample text with greedy decoding.

~80 MB RAM, ~10s on CPU
"""

import torch
import torch.nn as nn
import numpy as np
import time

# ── Data ──────────────────────────────────────────────────────────────────────
TEXT = (
    "shall i compare thee to a summers day "
    "thou art more lovely and more temperate "
    "rough winds do shake the darling buds of may "
    "and summers lease hath all too short a date"
)

chars   = sorted(set(TEXT))
vocab   = len(chars)
ch2ix   = {c: i for i, c in enumerate(chars)}
ix2ch   = {i: c for i, c in enumerate(chars)}
print(f"Vocabulary ({vocab} chars): {''.join(chars)}")
print(f"Text length: {len(TEXT)} characters\n")

# Encode full text
encoded = [ch2ix[c] for c in TEXT]

# Build (input, target) pairs: every consecutive pair of characters
inputs  = torch.tensor(encoded[:-1], dtype=torch.long)
targets = torch.tensor(encoded[1:],  dtype=torch.long)

# ── Model ─────────────────────────────────────────────────────────────────────
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed  = nn.Embedding(vocab_size, hidden_size)
        self.rnn    = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc     = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        emb    = self.embed(x)
        out, h = self.rnn(emb, h)
        logits = self.fc(out)
        return logits, h

    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size)


HIDDEN_SIZE = 32
EPOCHS      = 20
SEQ_CHUNK   = 40   # process text in chunks of 40 chars

torch.manual_seed(42)
model     = CharRNN(vocab, HIDDEN_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params}")

# ── Training ──────────────────────────────────────────────────────────────────
print("Training...")
start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    h = model.init_hidden(batch_size=1)
    total_loss = 0.0
    n_chunks   = 0

    for i in range(0, len(encoded) - SEQ_CHUNK, SEQ_CHUNK):
        chunk_in  = torch.tensor(encoded[i   : i+SEQ_CHUNK], dtype=torch.long).unsqueeze(0)  # [1, 40]
        chunk_tgt = torch.tensor(encoded[i+1 : i+SEQ_CHUNK+1], dtype=torch.long).unsqueeze(0) # [1, 40]

        h = h.detach()   # truncated BPTT — detach from previous chunk
        logits, h = model(chunk_in, h)
        loss = criterion(logits.view(-1, vocab), chunk_tgt.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
        n_chunks   += 1

    avg_loss = total_loss / max(n_chunks, 1)
    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:2d}/{EPOCHS} | Loss: {avg_loss:.4f}")

elapsed = time.time() - start
print(f"\nTraining complete in {elapsed:.1f}s")

# ── Text Generation ───────────────────────────────────────────────────────────
def generate(seed_char, length=60, temperature=1.0):
    model.eval()
    with torch.no_grad():
        h   = model.init_hidden(1)
        idx = ch2ix.get(seed_char, 0)
        out_chars = [seed_char]

        for _ in range(length):
            x      = torch.tensor([[idx]], dtype=torch.long)
            logits, h = model(x, h)
            logits = logits[0, 0] / temperature
            probs  = torch.softmax(logits, dim=0)
            idx    = torch.multinomial(probs, 1).item()
            out_chars.append(ix2ch[idx])

    return "".join(out_chars)

print("\n--- Generated Text Samples ---")
seeds = ['s', 't', 'r']
for seed in seeds:
    sample = generate(seed, length=60, temperature=0.8)
    print(f"  Seed '{seed}': {sample}")

print("\nDone! The RNN learned basic character patterns from the poem.")
