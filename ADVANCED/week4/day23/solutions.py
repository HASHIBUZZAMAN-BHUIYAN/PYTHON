# ~200 MB RAM, ~15s on CPU
"""
Day 23: Word Embeddings — Solutions
=====================================
Complete working solutions to all 5 exercises.
"""

import random
import numpy as np
from gensim.models import Word2Vec

random.seed(42)
np.random.seed(42)

# ─── Shared corpus ───────────────────────────────────────────────────────────
RAW_SENTENCES = [
    "the king ruled the kingdom with wisdom",
    "the queen sat beside the king on the throne",
    "the prince will become king one day",
    "the princess and the prince live in the castle",
    "a man walked into the castle to meet the king",
    "a woman met the queen in the royal garden",
    "the boy played with the girl in the park",
    "the dog chased the cat through the forest",
    "the lion is the king of the jungle",
    "the tiger and the lion are powerful animals",
    "the cat sat on the mat and slept",
    "the dog ran fast across the green field",
    "the scientist studied the stars in the sky",
    "the engineer built a new computer program",
    "software and hardware work together in a computer",
    "the algorithm processes data very efficiently",
    "paris is the capital city of france",
    "london is a famous city in england",
    "italy and france are countries in europe",
    "rome was once ruled by a powerful emperor",
    "the doctor studied medicine and biology",
    "the teacher explained mathematics to students",
    "students learn science physics and chemistry",
    "the chef cooked delicious food in the kitchen",
    "apples and bananas are healthy fruits",
    "the ocean is deep and full of fish",
    "the mountain is tall and covered in snow",
    "birds fly high above the forest and ocean",
    "the sun rises in the east every morning",
    "stars and planets orbit in outer space",
] * 20

CORPUS = [s.lower().split() for s in RAW_SENTENCES]

# Train model once, reuse in all solutions
print("Training Word2Vec model...")
MODEL = Word2Vec(
    sentences   = CORPUS,
    vector_size = 30,
    window      = 3,
    min_count   = 2,
    epochs      = 30,
    sg          = 1,
    seed        = 42,
    workers     = 1,
)
print(f"Model trained. Vocabulary: {len(MODEL.wv)} words\n")


# ─────────────────────────────────────────────
# Solution 1: Train and Inspect Word2Vec
# ─────────────────────────────────────────────
print("=" * 60)
print("SOLUTION 1: Train and Inspect Word2Vec")
print("=" * 60)

print(f"Vocabulary size: {len(MODEL.wv)}")

print("\nTop 5 similar to 'king':")
for w, s in MODEL.wv.most_similar("king", topn=5):
    print(f"  {w:<15} {s:.4f}")

print("\nTop 5 similar to 'dog':")
for w, s in MODEL.wv.most_similar("dog", topn=5):
    print(f"  {w:<15} {s:.4f}")

pairs = [("king", "queen"), ("dog", "cat"), ("king", "computer")]
print("\nPairwise similarities:")
for w1, w2 in pairs:
    sim = MODEL.wv.similarity(w1, w2)
    print(f"  similarity('{w1}', '{w2}') = {sim:.4f}")


# ─────────────────────────────────────────────
# Solution 2: Analogy Solver
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SOLUTION 2: Analogy Solver")
print("=" * 60)

analogies = [
    ("king",   "man",    "woman",  "queen"),
    ("prince", "man",    "woman",  "princess"),
    ("paris",  "france", "italy",  "rome"),
    ("dog",    "cat",    "lion",   "tiger"),
]

for a, b, c, expected in analogies:
    try:
        results = MODEL.wv.most_similar(positive=[a, c], negative=[b], topn=1)
        result, score = results[0]
        print(f"  {a} - {b} + {c} = {result}  (score: {score:.3f})  [expected: {expected}]")
    except KeyError as e:
        print(f"  Skipping — word not in vocabulary: {e}")


# ─────────────────────────────────────────────
# Solution 3: Cosine Similarity from Scratch
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SOLUTION 3: Cosine Similarity from Scratch")
print("=" * 60)

def cosine_similarity_numpy(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity using only numpy."""
    dot   = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

pairs = [("king", "queen"), ("man", "woman"),
         ("dog", "computer"), ("lion", "tiger"), ("ocean", "mountain")]

print(f"{'Pair':<25} {'From Scratch':>14} {'gensim.similarity':>18} {'Match?':>8}")
print("-" * 70)
for w1, w2 in pairs:
    try:
        v1      = MODEL.wv[w1]
        v2      = MODEL.wv[w2]
        scratch = cosine_similarity_numpy(v1, v2)
        gensim  = MODEL.wv.similarity(w1, w2)
        match   = abs(scratch - gensim) < 1e-5
        print(f"  ({w1}, {w2}){'':<{20-len(w1)-len(w2)}} {scratch:>14.6f} {gensim:>18.6f} {'✓' if match else '✗':>8}")
    except KeyError as e:
        print(f"  ({w1}, {w2}) — skipped, word not in vocab: {e}")


# ─────────────────────────────────────────────
# Solution 4: Document Embedding via Word Averaging
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SOLUTION 4: Document Embedding via Word Averaging")
print("=" * 60)

def embed_document(model: Word2Vec, text: str) -> np.ndarray:
    """Compute document embedding as mean of word vectors."""
    tokens = text.lower().split()
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if not vectors:
        return np.zeros(model.wv.vector_size)
    return np.mean(vectors, axis=0)

docs = [
    "the king and queen rule the kingdom",
    "the prince and princess live in the castle",
    "the dog and cat are friendly animals",
    "the lion and tiger are wild animals",
    "computers and software run algorithms",
]

embeddings = [embed_document(MODEL, doc) for doc in docs]
print("Document embeddings (first 5 dims):")
for i, (doc, emb) in enumerate(zip(docs, embeddings)):
    print(f"  Doc {i}: '{doc[:35]}' → [{', '.join(f'{v:.3f}' for v in emb[:5])}...]")

print(f"\nSimilarity of all docs to Doc 0 ('{docs[0][:30]}...'):")
ref = embeddings[0]
for i, (doc, emb) in enumerate(zip(docs, embeddings)):
    sim = cosine_similarity_numpy(ref, emb)
    print(f"  Doc {i}: {sim:.4f}  '{doc[:45]}'")

print("\nDoc 1 (royalty) should be most similar to Doc 0 (royalty).")


# ─────────────────────────────────────────────
# Solution 5: Odd-One-Out Detector
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SOLUTION 5: Odd-One-Out Detector")
print("=" * 60)

groups = [
    ["king", "queen", "prince", "dog"],
    ["cat", "dog", "lion", "computer"],
    ["paris", "london", "rome", "tiger"],
    ["man", "woman", "boy", "ocean"],
]

for group in groups:
    # Filter to words in vocab
    valid = [w for w in group if w in MODEL.wv]
    missing = [w for w in group if w not in MODEL.wv]

    if missing:
        print(f"  Warning: words not in vocab: {missing}")

    if len(valid) < 3:
        print(f"  Group {group}: too few valid words — skipping")
        continue

    try:
        odd = MODEL.wv.doesnt_match(valid)
        print(f"  Group: {valid} → Odd one out: '{odd}'")
    except Exception as e:
        print(f"  Group {group}: error — {e}")

print("\n--- Day 23 Solutions Complete ---")
