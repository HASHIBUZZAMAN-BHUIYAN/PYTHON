# ~200 MB RAM, ~15s on CPU
"""
Project 1: Similar Words Tool
================================
What it does: Trains a Word2Vec model on a tech/science corpus and provides
              a "find similar words" lookup tool. Demonstrates how word
              embeddings capture semantic relationships.

What it teaches:
  - Training gensim Word2Vec on domain-specific text
  - Using most_similar() to explore embedding neighborhoods
  - How domain affects which words become close in vector space
  - Interpreting cosine similarity scores
"""

import random
import numpy as np
from gensim.models import Word2Vec

random.seed(42)
np.random.seed(42)

# ─── Tech/Science Corpus (hardcoded sentences) ───────────────────────────────

CORPUS_SENTENCES = [
    # Programming
    "python is a popular programming language used for data science",
    "javascript is used for web development and frontend programming",
    "machine learning algorithms are implemented in python and java",
    "the programmer wrote code to solve the algorithm problem",
    "debugging code requires patience and systematic thinking",
    "functions and classes are fundamental concepts in programming",
    "arrays and lists store multiple values in computer memory",
    "loops and recursion are used to repeat code in programming",
    "variables store data values that can change during execution",
    "software engineers write test code to verify program correctness",
    # Data Science
    "data science combines statistics mathematics and programming",
    "pandas and numpy are essential libraries for data analysis",
    "machine learning models learn patterns from training data",
    "neural networks process data through multiple connected layers",
    "deep learning requires large datasets and significant computation",
    "data preprocessing cleans and transforms raw data for analysis",
    "feature engineering creates useful inputs for machine learning",
    "cross validation evaluates model performance on unseen data",
    "overfitting occurs when a model memorizes training data too well",
    "regularization prevents overfitting by penalizing model complexity",
    # Artificial Intelligence
    "artificial intelligence simulates human intelligence in machines",
    "natural language processing enables computers to understand text",
    "computer vision allows machines to interpret and analyze images",
    "reinforcement learning trains agents through reward and punishment",
    "chatbots use natural language processing to converse with users",
    "recommendation systems suggest products based on user behavior",
    "speech recognition converts spoken audio to written text",
    "image recognition classifies objects within photographs",
    "autonomous vehicles use sensors and algorithms to navigate roads",
    "robots are programmed with artificial intelligence to perform tasks",
    # Hardware & Systems
    "processors execute billions of instructions every second",
    "memory stores temporary data while programs are running",
    "graphics processing units accelerate parallel computations",
    "solid state drives store data faster than traditional hard drives",
    "networks connect computers to share data and resources",
    "cloud computing provides remote servers for storage and processing",
    "operating systems manage hardware and software resources",
    "databases store organize and retrieve structured data efficiently",
    "encryption protects data by converting it into unreadable format",
    "internet of things connects everyday devices to the internet",
    # Mathematics & Science
    "statistics analyzes and interprets numerical data patterns",
    "probability measures the likelihood of events occurring",
    "linear algebra provides tools for machine learning mathematics",
    "calculus is essential for understanding neural network training",
    "physics describes the laws that govern the natural universe",
    "chemistry studies the composition and properties of matter",
    "biology explores the structure and function of living organisms",
    "mathematics is the foundation of all scientific disciplines",
    "experiments test hypotheses through controlled observations",
    "scientific research advances human knowledge and technology",
]

# Augment corpus by repeating with slight shuffling
augmented = CORPUS_SENTENCES.copy()
for _ in range(8):
    shuffled = CORPUS_SENTENCES[:]
    random.shuffle(shuffled)
    augmented.extend(shuffled)

CORPUS = [s.lower().split() for s in augmented]

# ─── Train Word2Vec ──────────────────────────────────────────────────────────
print("=" * 60)
print("SIMILAR WORDS TOOL — Tech/Science Word2Vec")
print("=" * 60)
print(f"Corpus: {len(CORPUS)} sentences")

model = Word2Vec(
    sentences   = CORPUS,
    vector_size = 50,
    window      = 4,
    min_count   = 3,
    epochs      = 80,
    sg          = 1,
    seed        = 42,
    workers     = 1,
)
print(f"Vocabulary: {len(model.wv)} words")
print(f"Embedding dimension: {model.wv.vector_size}")

# ─── Similar Words Lookup ────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("TOP-5 SIMILAR WORDS FOR QUERY TERMS")
print("─" * 60)

query_words = [
    "python",
    "machine",
    "data",
    "algorithm",
    "neural",
    "programming",
    "learning",
    "science",
]

for query in query_words:
    if query not in model.wv:
        print(f"\n'{query}' not in vocabulary — skipping")
        continue

    similar = model.wv.most_similar(query, topn=5)
    print(f"\nTop-5 similar to '{query}':")
    for rank, (word, score) in enumerate(similar, 1):
        bar = "█" * int(score * 30)
        print(f"  {rank}. {word:<20} {score:.4f}  {bar}")

# ─── Semantic Clusters ───────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("SEMANTIC NEIGHBORHOOD EXPLORATION")
print("─" * 60)

seed_words = ["learning", "data", "code"]
for seed in seed_words:
    if seed not in model.wv:
        continue
    neighbors = model.wv.most_similar(seed, topn=10)
    neighbor_words = [w for w, _ in neighbors]
    print(f"\nNeighborhood of '{seed}':")
    print(f"  {neighbor_words}")

# ─── Distance Comparisons ────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("DISTANCE COMPARISONS")
print("─" * 60)

comparison_pairs = [
    ("python",    "java"),
    ("machine",   "learning"),
    ("data",      "science"),
    ("neural",    "network"),
    ("python",    "chemistry"),
    ("algorithm", "biology"),
]

print(f"{'Pair':<30} {'Similarity':>12} {'Relationship'}")
print("-" * 60)
for w1, w2 in comparison_pairs:
    if w1 in model.wv and w2 in model.wv:
        sim = model.wv.similarity(w1, w2)
        rel = "related" if sim > 0.5 else ("somewhat related" if sim > 0.3 else "unrelated")
        print(f"  ({w1}, {w2}){'':<{25-len(w1)-len(w2)}} {sim:>12.4f}  {rel}")
    else:
        missing = [w for w in [w1, w2] if w not in model.wv]
        print(f"  ({w1}, {w2}) — not in vocab: {missing}")

print("\n--- Project 1 Complete ---")
