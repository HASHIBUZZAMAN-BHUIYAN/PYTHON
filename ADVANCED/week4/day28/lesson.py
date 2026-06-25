# Advanced Day 28 — NLP Capstone: Retrieval Chatbot
# ~40 MB RAM, ~3s on CPU

print("""
=== NLP Week Capstone — Retrieval Chatbot — Day 28 ===

A retrieval-based chatbot works by:
  1. Storing a knowledge base of (question, answer) pairs
  2. Encoding every question as a TF-IDF vector
  3. When a user asks something, encode the query and find the
     most similar stored question using cosine similarity
  4. Return the answer paired with the best match
""")

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── KNOWLEDGE BASE ───────────────────────────────────────────────────────────
KB = [
    ("What is machine learning?",
     "Machine learning is a type of AI where systems learn patterns from data without being explicitly programmed."),
    ("How does deep learning work?",
     "Deep learning uses multi-layer neural networks to automatically learn hierarchical features from data."),
    ("What is a neural network?",
     "A neural network is a computational model inspired by the brain, made of interconnected layers of nodes."),
    ("What is Python?",
     "Python is a high-level, general-purpose programming language known for readability and versatility."),
    ("What is overfitting?",
     "Overfitting occurs when a model learns the training data too well and fails to generalize to new data."),
    ("What is the difference between supervised and unsupervised learning?",
     "Supervised learning uses labeled data. Unsupervised learning finds patterns in unlabeled data."),
    ("What is gradient descent?",
     "Gradient descent is an optimization algorithm that adjusts model weights to minimize the loss function."),
    ("What is a transformer in NLP?",
     "Transformers are neural network architectures using self-attention to process sequential data in parallel."),
    ("What is NLP?",
     "NLP (Natural Language Processing) is the field of AI that enables computers to understand human language."),
    ("What is regularization?",
     "Regularization adds a penalty term to the loss to prevent overfitting by keeping model weights small."),
]

def preprocess(text):
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

questions = [preprocess(q) for q, _ in KB]
answers   = [a for _, a in KB]

# Build TF-IDF index
vec = TfidfVectorizer(ngram_range=(1,2), max_features=500)
Q_matrix = vec.fit_transform(questions)

def chat(query, threshold=0.1):
    q_vec = vec.transform([preprocess(query)])
    sims  = cosine_similarity(q_vec, Q_matrix).flatten()
    best  = int(np.argmax(sims))
    if sims[best] < threshold:
        return "I don't know the answer to that. Try rephrasing your question.", 0.0
    return answers[best], float(sims[best])

# ─── DEMO ─────────────────────────────────────────────────────────────────────
print("=== Chatbot Demo ===")
TEST_QUERIES = [
    "Can you explain what machine learning is?",
    "How do neural networks learn?",
    "Tell me about Python programming",
    "What happens when a model memorizes training data?",
    "Explain the attention mechanism in transformers",
    "What is the best pizza topping?",  # Out of domain
]

print(f"\n{'Query':<50}  {'Confidence':>10}")
print("-"*70)
for query in TEST_QUERIES:
    answer, conf = chat(query)
    print(f"{query:<50}  {conf:>10.3f}")
    print(f"  → {answer[:80]}")
    print()

# ─── IMPROVEMENTS OVERVIEW ────────────────────────────────────────────────────
print("""
=== Design Decisions ===

1. Preprocessing  — lowercase + remove punctuation improves matching
2. TF-IDF         — weights rare words more; ngrams help multi-word queries
3. Cosine sim     — length-normalized; works better than dot product
4. Threshold      — rejects off-topic queries cleanly
5. Confidence log — helps debug and improve the knowledge base

Limits of retrieval chatbots:
  - Can only answer questions in the knowledge base
  - Sensitive to vocabulary mismatch (ask "gradient descent" ≠ "optimize weights")
  - TF-IDF does not understand meaning, only word overlap
  → Solution: replace TF-IDF with sentence embeddings (Lesson Day 23)
""")
