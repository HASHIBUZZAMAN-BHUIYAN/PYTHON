# Advanced Day 27 — Text Generation & Summarization
# ~40 MB RAM, ~3s on CPU

print("""
=== Text Generation & Summarization — Day 27 ===

Text Generation: create new text that resembles a training corpus.
Summarization:   compress long text into a shorter version.

Approaches:
  Generation:   Markov chains (bigram/trigram), template-based
  Summarization: Extractive (TF-IDF sentence scoring), Abstractive (T5/BART)
""")

import re
import random
from collections import defaultdict

# ─── TRAINING CORPUS ─────────────────────────────────────────────────────────
CORPUS_TEXT = """
The quick brown fox jumps over the lazy dog.
The dog barked loudly at the fox in the meadow.
A fox is a clever animal that hunts at night.
The meadow was full of flowers and tall green grass.
Brown bears often hunt for fish in the mountain stream.
The stream flows quickly through the dense forest.
Dense forests are home to many wild animals and birds.
Birds sing in the morning when the sun rises over the hills.
The hills are covered in snow during the cold winter months.
Winter brings cold winds and frost to the mountain valleys.
"""

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()

tokens = preprocess(CORPUS_TEXT)

# ─── 1. BIGRAM MARKOV CHAIN ──────────────────────────────────────────────────
print("=== 1. Bigram Markov Chain ===")
bigram_model = defaultdict(list)
for i in range(len(tokens)-1):
    bigram_model[tokens[i]].append(tokens[i+1])

def generate_bigram(start_word, n_words=20, seed=42):
    random.seed(seed)
    word = start_word.lower()
    result = [word]
    for _ in range(n_words - 1):
        nexts = bigram_model.get(word, [])
        if not nexts: break
        word = random.choice(nexts)
        result.append(word)
    return " ".join(result)

for seed, start in [(0,"the"),(1,"a"),(2,"brown")]:
    print(f"  Start='{start}': {generate_bigram(start, n_words=15, seed=seed)}")

# ─── 2. TRIGRAM MARKOV CHAIN ─────────────────────────────────────────────────
print("\n=== 2. Trigram Markov Chain ===")
trigram_model = defaultdict(list)
for i in range(len(tokens)-2):
    key = (tokens[i], tokens[i+1])
    trigram_model[key].append(tokens[i+2])

def generate_trigram(start_pair, n_words=20, seed=42):
    random.seed(seed)
    w1, w2 = start_pair
    result = [w1, w2]
    for _ in range(n_words - 2):
        nexts = trigram_model.get((w1, w2), [])
        if not nexts: break
        w3 = random.choice(nexts)
        result.append(w3)
        w1, w2 = w2, w3
    return " ".join(result)

for seed, pair in [(0,("the","quick")),(1,("the","meadow")),(2,("dense","forests"))]:
    print(f"  Start={pair}: {generate_trigram(pair, n_words=15, seed=seed)}")

# ─── 3. EXTRACTIVE SUMMARIZATION ─────────────────────────────────────────────
print("\n=== 3. Extractive Summarization (TF-IDF) ===")
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

ARTICLE = """
Researchers at MIT have developed a new battery technology that could transform
electric vehicles and grid storage. The lithium-sulfur battery achieves energy
density three times higher than conventional lithium-ion cells while using abundant
and cheap sulfur as the cathode material. The key breakthrough involved coating
sulfur particles with a conductive polymer that prevents the degradation typically
seen in sulfur cathodes. Tests showed the battery retains 80 percent capacity
after 500 charge cycles. Commercial applications could include electric trucks
and long-haul aviation. The team expects to begin pilot manufacturing within
two years. Industry analysts estimate the technology could reduce battery costs
by 60 percent compared to current lithium-ion prices.
"""

def extractive_summarize(text, n_sentences=3):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.strip()) > 10]
    if len(sents) <= n_sentences: return " ".join(sents)
    vec   = TfidfVectorizer(stop_words="english")
    X     = vec.fit_transform(sents)
    scores= X.sum(axis=1).A1
    top   = np.argsort(scores)[-n_sentences:]
    top   = sorted(top)
    return " ".join(sents[i] for i in top)

summary = extractive_summarize(ARTICLE, n_sentences=3)
orig_words = len(ARTICLE.split())
summ_words = len(summary.split())
print(f"  Original: {orig_words} words")
print(f"  Summary ({summ_words} words):\n    {summary}")

# ─── 4. TEMPLATE-BASED GENERATION ────────────────────────────────────────────
print("\n=== 4. Template-Based Generation ===")
TEMPLATES = [
    "Scientists at {org} discovered that {subject} can {verb} {object} by {percent}%.",
    "A new study shows {subject} {verb} {object} in {timeframe}.",
    "Researchers found that {subject} significantly {verb} the risk of {condition}.",
]
SLOTS = {
    "org":      ["MIT","Stanford","Oxford","NASA"],
    "subject":  ["exercise","sleep","meditation","diet"],
    "verb":     ["reduce","improve","enhance","double"],
    "object":   ["memory","stress","energy","recovery"],
    "percent":  ["20","35","50","42"],
    "timeframe":["two weeks","30 days","six months"],
    "condition":["diabetes","depression","heart disease"],
}
random.seed(7)
for t in TEMPLATES:
    for slot, choices in SLOTS.items():
        t = t.replace("{"+slot+"}", random.choice(choices))
    print(f"  {t}")

print("""
=== Summary ===
  Markov chains — fast, fun, requires large corpus for quality output
  Extractive    — reliable, preserves original language, no model download
  Abstractive   — best quality (T5/BART), but ~600MB model, slower
  Template      — guaranteed coherent output; limited variety
""")
