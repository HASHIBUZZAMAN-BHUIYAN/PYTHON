"""
Day 30 Project 3: Sentiment Classification — LSTM vs TF-IDF+LR Baseline
=========================================================================
200 synthetic short reviews (100 positive, 100 negative).
Build vocab, encode sequences, pad to max_len=20.
LSTM hidden=32 for classification.
Compare to TF-IDF + Logistic Regression baseline.
Print accuracy for both methods.

~100 MB RAM, ~15s on CPU
"""

import torch
import torch.nn as nn
import numpy as np
import time
from collections import Counter

# ── Synthetic Dataset ─────────────────────────────────────────────────────────
POS_WORDS = ["great", "excellent", "amazing", "love", "fantastic", "good",
             "wonderful", "brilliant", "superb", "perfect", "best", "enjoy"]
NEG_WORDS = ["terrible", "awful", "horrible", "hate", "bad", "worst",
             "disappointing", "poor", "dreadful", "dislike", "boring", "waste"]
NEUTRAL   = ["the", "this", "was", "very", "product", "movie", "book",
             "film", "item", "experience", "quite", "really", "it", "and"]

np.random.seed(42)

def make_review(sentiment, n_words=8):
    words = []
    if sentiment == 1:
        n_pos = np.random.randint(2, 4)
        words += list(np.random.choice(POS_WORDS, n_pos, replace=False))
    else:
        n_neg = np.random.randint(2, 4)
        words += list(np.random.choice(NEG_WORDS, n_neg, replace=False))
    n_neutral = n_words - len(words)
    words += list(np.random.choice(NEUTRAL, n_neutral, replace=True))
    np.random.shuffle(words)
    return " ".join(words)

reviews = [(make_review(1), 1) for _ in range(100)] + \
          [(make_review(0), 0) for _ in range(100)]
np.random.shuffle(reviews)
texts, labels = zip(*reviews)
labels = list(labels)

print(f"Dataset: {len(texts)} reviews (100 pos, 100 neg)")
print(f"Example positive: '{texts[labels.index(1)]}'")
print(f"Example negative: '{texts[labels.index(0)]}'")

# ── Train/Test Split ──────────────────────────────────────────────────────────
split   = 160   # 80/20
train_texts, test_texts   = texts[:split], texts[split:]
train_labels, test_labels = labels[:split], labels[split:]

# ─────────────────────────────────────────────────────────────────────────────
# APPROACH 1: TF-IDF + Logistic Regression (Baseline)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("APPROACH 1: TF-IDF + Logistic Regression (Baseline)")
print("─" * 50)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    t0  = time.time()
    tfidf = TfidfVectorizer(max_features=100)
    X_tr_tfidf = tfidf.fit_transform(train_texts)
    X_te_tfidf = tfidf.transform(test_texts)

    lr = LogisticRegression(max_iter=200, random_state=42)
    lr.fit(X_tr_tfidf, train_labels)
    lr_pred = lr.predict(X_te_tfidf)
    lr_acc  = accuracy_score(test_labels, lr_pred)
    lr_time = time.time() - t0
    print(f"  TF-IDF+LR Accuracy: {lr_acc:.2%}  (in {lr_time:.2f}s)")
    BASELINE_ACC = lr_acc
except ImportError:
    print("  sklearn not available — skipping baseline")
    BASELINE_ACC = None

# ─────────────────────────────────────────────────────────────────────────────
# APPROACH 2: LSTM Classifier
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("APPROACH 2: LSTM Classifier")
print("─" * 50)

# Build vocabulary from training set
MAX_LEN = 20
PAD_IDX = 0
UNK_IDX = 1

all_train_words = " ".join(train_texts).split()
word_counts = Counter(all_train_words)
vocab_words = ["<PAD>", "<UNK>"] + [w for w, c in word_counts.most_common(200)]
word2idx    = {w: i for i, w in enumerate(vocab_words)}
VOCAB_SIZE  = len(vocab_words)
print(f"  Vocab size: {VOCAB_SIZE}")

def encode_and_pad(text, word2idx, max_len):
    tokens = text.split()[:max_len]
    ids    = [word2idx.get(t, UNK_IDX) for t in tokens]
    padded = ids + [PAD_IDX] * (max_len - len(ids))
    return padded

def prepare_data(texts, labels, word2idx, max_len):
    X = torch.tensor([encode_and_pad(t, word2idx, max_len) for t in texts],
                     dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y

X_tr, y_tr = prepare_data(train_texts, train_labels, word2idx, MAX_LEN)
X_te, y_te = prepare_data(test_texts,  test_labels,  word2idx, MAX_LEN)

# ── LSTM Model ────────────────────────────────────────────────────────────────
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_size=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm  = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc    = nn.Linear(hidden_size, 2)
        self.drop  = nn.Dropout(0.3)

    def forward(self, x):
        emb       = self.drop(self.embed(x))
        _, (h_n, _) = self.lstm(emb)
        return self.fc(self.drop(h_n.squeeze(0)))

torch.manual_seed(42)
model_lstm = SentimentLSTM(VOCAB_SIZE, embed_dim=16, hidden_size=32)
optimizer  = torch.optim.Adam(model_lstm.parameters(), lr=0.005)
criterion  = nn.CrossEntropyLoss()

n_params = sum(p.numel() for p in model_lstm.parameters())
print(f"  LSTM parameters: {n_params}")

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS     = 25
BATCH_SIZE = 32
t0         = time.time()

for epoch in range(1, EPOCHS + 1):
    model_lstm.train()
    perm       = torch.randperm(len(X_tr))
    total_loss = 0.0
    n_batches  = 0

    for i in range(0, len(X_tr), BATCH_SIZE):
        idx    = perm[i : i+BATCH_SIZE]
        xb, yb = X_tr[idx], y_tr[idx]
        logits = model_lstm(xb)
        loss   = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    if epoch % 5 == 0 or epoch == 1:
        model_lstm.eval()
        with torch.no_grad():
            tr_acc = (model_lstm(X_tr).argmax(1) == y_tr).float().mean().item()
        print(f"  Epoch {epoch:2d} | Loss: {total_loss/n_batches:.4f} | TrainAcc: {tr_acc:.2%}")

lstm_time = time.time() - t0
print(f"\nLSTM training complete in {lstm_time:.1f}s")

# ── Evaluation ────────────────────────────────────────────────────────────────
model_lstm.eval()
with torch.no_grad():
    lstm_pred = model_lstm(X_te).argmax(1)
    lstm_acc  = (lstm_pred == y_te).float().mean().item()

# ── Results Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("RESULTS COMPARISON")
print("=" * 50)
print(f"  {'Method':<25} {'Accuracy':>10}")
print(f"  {'-'*36}")
if BASELINE_ACC is not None:
    print(f"  {'TF-IDF + Logistic Reg':<25} {BASELINE_ACC:>9.2%}")
print(f"  {'LSTM (hidden=32)':<25} {lstm_acc:>9.2%}")

if BASELINE_ACC is not None:
    diff = lstm_acc - BASELINE_ACC
    winner = "LSTM" if diff > 0 else "TF-IDF+LR"
    print(f"\n  Winner: {winner} (diff: {diff:+.2%})")
    print("\nNote: On small datasets (200 samples), simple baselines often")
    print("      match or beat LSTMs. LSTMs shine on larger, complex data.")
