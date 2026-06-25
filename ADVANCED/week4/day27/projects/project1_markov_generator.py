"""
Project: Markov Chain Text Generator
Teaches: bigram/trigram Markov models, temperature sampling, building a
         small "language model" from scratch with zero dependencies.
~10 MB RAM, <1s on CPU
"""
import re, random
import numpy as np
from collections import defaultdict

TRAINING_TEXT = """
In the beginning the universe was created. This made a lot of people very angry
and has been widely regarded as a bad move. The ships hung in the sky in much
the same way that bricks don't. Far out in the uncharted backwaters of the
unfashionable end of the western spiral arm of the Galaxy lies a small unregarded
yellow sun. Orbiting this at a distance of roughly ninety-two million miles is an
utterly insignificant little blue green planet whose ape-descended life forms are
so amazingly primitive that they still think digital watches are a pretty neat idea.
The planet has a problem which was this. Most of the people living on it were
unhappy for pretty much of the time. Many solutions were suggested for this problem
but most of these were largely concerned with the movements of small green pieces
of paper which is odd because on the whole it was not the small green pieces of
paper that were unhappy. And so the problem remained and lots of the people were
mean and most of them were miserable even the ones with digital watches.
"""

tokens = re.sub(r"[^a-z\s]", "", TRAINING_TEXT.lower()).split()

# ─── Build models ─────────────────────────────────────────────────────────────
def build_bigram(toks):
    m = defaultdict(list)
    for i in range(len(toks)-1):
        m[toks[i]].append(toks[i+1])
    return m

def build_trigram(toks):
    m = defaultdict(list)
    for i in range(len(toks)-2):
        m[(toks[i], toks[i+1])].append(toks[i+2])
    return m

bigram  = build_bigram(tokens)
trigram = build_trigram(tokens)

# ─── Temperature-aware generator ─────────────────────────────────────────────
def gen_with_temp(model_type, start, length=30, temperature=1.0, seed=42):
    np.random.seed(seed); random.seed(seed)

    if model_type == "bigram":
        result = [start]
        current = start
        for _ in range(length - 1):
            candidates = bigram.get(current, list(bigram.keys()))
            unique = list(set(candidates))
            counts = np.array([candidates.count(w) for w in unique], dtype=float)
            logp   = np.log(counts + 1e-10) / temperature
            logp  -= logp.max()
            probs  = np.exp(logp); probs /= probs.sum()
            current = np.random.choice(unique, p=probs)
            result.append(current)
    else:  # trigram
        words  = start.split()[:2]
        result = words[:]
        for _ in range(length - 2):
            key  = (result[-2], result[-1])
            cands= trigram.get(key, list(tokens))
            unique = list(set(cands))
            counts = np.array([cands.count(w) for w in unique], dtype=float)
            logp   = np.log(counts + 1e-10) / temperature
            logp  -= logp.max()
            probs  = np.exp(logp); probs /= probs.sum()
            result.append(np.random.choice(unique, p=probs))
    return " ".join(result)

print("=== Markov Text Generator ===\n")
print("--- Bigram at different temperatures ---")
for temp in [0.2, 1.0, 3.0]:
    text = gen_with_temp("bigram", "the", length=20, temperature=temp)
    print(f"  temp={temp:.1f}: {text}")

print("\n--- Trigram generation ---")
for seed in [0, 1, 2]:
    text = gen_with_temp("trigram", "the planet", length=20, temperature=1.0, seed=seed)
    print(f"  seed={seed}: {text}")

# ─── Word frequency in generated text ─────────────────────────────────────────
from collections import Counter
generated_big = gen_with_temp("bigram", "the", length=200, temperature=1.0)
freq = Counter(generated_big.split()).most_common(10)
print(f"\n  Top words in generated text: {freq}")

# ─── Perplexity proxy (log-likelihood of training text under bigram) ──────────
def log_likelihood(toks, model):
    ll = 0; n = 0
    for i in range(len(toks)-1):
        cands = model.get(toks[i], [])
        if not cands: continue
        count_next = cands.count(toks[i+1])
        prob = (count_next + 1e-9) / (len(cands) + 1e-9)
        ll  += np.log(prob); n += 1
    return ll / max(n, 1)

ll = log_likelihood(tokens, bigram)
print(f"\n  Avg log-likelihood on training text: {ll:.4f}  (higher = better fit)")
