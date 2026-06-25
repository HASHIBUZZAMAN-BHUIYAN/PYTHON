"""
Day 30 Project 1: Character-Level Text Generator with LSTM
===========================================================
Train an LSTM to generate text character by character.
Train on a hardcoded ~300-character paragraph.
Architecture: Embedding(vocab→16) → LSTM(hidden=64) → Linear(vocab)
Training: 30 epochs
Output: 3 generated samples of 50 chars each.

~100 MB RAM, ~20s on CPU
"""

import torch
import torch.nn as nn
import numpy as np
import time

torch.manual_seed(42)

# ── Training Text ─────────────────────────────────────────────────────────────
TEXT = (
    "to be or not to be that is the question "
    "whether tis nobler in the mind to suffer "
    "the slings and arrows of outrageous fortune "
    "or to take arms against a sea of troubles "
    "and by opposing end them to die to sleep "
    "no more and by a sleep to say we end "
    "the heartache and the thousand natural shocks"
)

# ── Vocabulary ────────────────────────────────────────────────────────────────
chars  = sorted(set(TEXT))
vocab  = len(chars)
ch2ix  = {c: i for i, c in enumerate(chars)}
ix2ch  = {i: c for i, c in enumerate(chars)}
encoded = [ch2ix[c] for c in TEXT]

print(f"Text length:    {len(TEXT)} chars")
print(f"Vocabulary:     {vocab} unique chars: {''.join(chars)}")

# ── Build Training Sequences ──────────────────────────────────────────────────
SEQ_LEN = 30   # each training chunk is 30 chars → predict next 30

def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(0, len(data) - seq_len - 1, seq_len // 2):  # stride = seq_len/2
        X.append(data[i     : i+seq_len])
        y.append(data[i+1   : i+seq_len+1])
    return X, y

seqs_X, seqs_y = make_sequences(encoded, SEQ_LEN)
X_train = torch.tensor(seqs_X, dtype=torch.long)   # [N, 30]
y_train = torch.tensor(seqs_y, dtype=torch.long)   # [N, 30]
print(f"Training sequences: {X_train.shape[0]}\n")

# ── Model ─────────────────────────────────────────────────────────────────────
EMBED_DIM   = 16
HIDDEN_SIZE = 64

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm  = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc    = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, state=None):
        emb        = self.embed(x)         # [batch, seq, embed]
        out, state = self.lstm(emb, state) # out: [batch, seq, hidden]
        logits     = self.fc(out)          # [batch, seq, vocab]
        return logits, state

    def init_state(self, batch_size=1):
        h = torch.zeros(1, batch_size, self.hidden_size)
        c = torch.zeros(1, batch_size, self.hidden_size)
        return (h, c)


model = CharLSTM(vocab, EMBED_DIM, HIDDEN_SIZE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params}")

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS     = 30
BATCH_SIZE = 16

print("Training LSTM character model...")
start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(X_train))
    total_loss = 0.0
    n_batches  = 0

    for i in range(0, len(X_train), BATCH_SIZE):
        idx = perm[i : i+BATCH_SIZE]
        xb  = X_train[idx]              # [batch, seq_len]
        yb  = y_train[idx]              # [batch, seq_len]

        logits, _ = model(xb)           # [batch, seq_len, vocab]
        # reshape for loss: [batch*seq, vocab] vs [batch*seq]
        loss = criterion(logits.view(-1, vocab), yb.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:2d}/{EPOCHS} | Loss: {total_loss/n_batches:.4f}")

elapsed = time.time() - start
print(f"\nTraining complete in {elapsed:.1f}s")

# ── Text Generation ───────────────────────────────────────────────────────────
def generate_text(model, seed_char, length=50, temperature=0.8):
    model.eval()
    with torch.no_grad():
        state = model.init_state(1)
        idx   = ch2ix.get(seed_char, 0)
        result = [seed_char]

        for _ in range(length):
            x      = torch.tensor([[idx]], dtype=torch.long)
            logits, state = model(x, state)
            logits = logits[0, 0] / temperature
            probs  = torch.softmax(logits, dim=0)
            idx    = torch.multinomial(probs, 1).item()
            result.append(ix2ch[idx])

    return "".join(result)

print("\n--- Generated Text Samples (50 chars each) ---")
seeds = ['t', 'a', 'w']
for seed in seeds:
    sample = generate_text(model, seed, length=50, temperature=0.8)
    print(f"  Seed '{seed}': \"{sample}\"")

print("\nNote: with only 30 epochs on ~300 chars, output is approximate.")
print("      The LSTM picks up common bigrams/trigrams from the text.")
