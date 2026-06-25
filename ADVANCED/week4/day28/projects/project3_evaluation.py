"""
Project: Chatbot Evaluation — MRR, Hit@K, and Confidence Analysis
Teaches: quantitatively evaluating a retrieval chatbot's quality using
         MRR (Mean Reciprocal Rank), Hit@1, Hit@3 metrics.
~25 MB RAM, ~1s on CPU
"""
import re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── KB ───────────────────────────────────────────────────────────────────────
KB = [
    ("what is machine learning",   0),
    ("what is deep learning",      1),
    ("what is a neural network",   2),
    ("what is python",             3),
    ("what is overfitting",        4),
    ("what is supervised learning",5),
    ("what is gradient descent",   6),
    ("what is a transformer",      7),
    ("what is natural language processing", 8),
    ("what is regularization",     9),
]
ANSWERS = [
    "ML systems learn from data.",
    "Deep learning uses multi-layer neural networks.",
    "Neural nets are interconnected computational nodes.",
    "Python is a versatile programming language.",
    "Overfitting: model memorizes training data.",
    "Supervised learning uses labeled data.",
    "Gradient descent minimizes loss iteratively.",
    "Transformers use self-attention mechanisms.",
    "NLP enables computers to understand text.",
    "Regularization prevents overfitting via weight penalties.",
]

def preprocess(t): return re.sub(r"[^\w\s]","",t.lower()).strip()
questions = [preprocess(q) for q,_ in KB]

vec  = TfidfVectorizer(ngram_range=(1,2), max_features=500)
Qmat = vec.fit_transform(questions)

def retrieve_ranked(query, k=5):
    q_vec = vec.transform([preprocess(query)])
    sims  = cosine_similarity(q_vec, Qmat).flatten()
    ranked= np.argsort(sims)[::-1]
    return ranked[:k], sims[ranked[:k]]

# ─── Evaluation dataset: paraphrased queries → expected KB index ─────────────
EVAL = [
    ("Can you explain machine learning to me?",          0),
    ("How does deep learning actually work?",             1),
    ("Tell me about neural networks.",                    2),
    ("What programming language is Python?",              3),
    ("My model memorizes training data, what is that?",  4),
    ("What is the difference between supervised approaches?", 5),
    ("How to optimize a neural network using gradients?", 6),
    ("Explain the transformer attention mechanism.",      7),
    ("How do computers process human language?",         8),
    ("How do we prevent overfitting with weight penalty?",9),
]

print("=== Chatbot Evaluation ===\n")

hits_at_1, hits_at_3, reciprocal_ranks = [], [], []
print(f"{'Query':<50}  {'Rank':>5}  {'Score':>7}")
print("─"*70)
for query, expected_idx in EVAL:
    ranked, scores = retrieve_ranked(query, k=len(KB))
    rank = None
    for r, idx in enumerate(ranked, start=1):
        if idx == expected_idx:
            rank = r; break
    rank = rank or len(KB) + 1
    hits_at_1.append(1 if rank <= 1 else 0)
    hits_at_3.append(1 if rank <= 3 else 0)
    reciprocal_ranks.append(1.0 / rank)
    print(f"  {query:<48}  {rank:>5}  {scores[0]:>7.3f}")

mrr     = np.mean(reciprocal_ranks)
hit_at1 = np.mean(hits_at_1)
hit_at3 = np.mean(hits_at_3)

print(f"\n{'─'*50}")
print(f"  MRR    (Mean Reciprocal Rank): {mrr:.3f}")
print(f"  Hit@1  (Exact top-1 accuracy): {hit_at1:.1%}")
print(f"  Hit@3  (Answer in top 3):      {hit_at3:.1%}")

# ─── Visualize ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
metrics   = ["MRR", "Hit@1", "Hit@3"]
values    = [mrr, hit_at1, hit_at3]
axes[0].bar(metrics, values, color=["steelblue","tomato","limegreen"],
            alpha=0.85, edgecolor="k", linewidth=0.7)
for i, v in enumerate(values):
    axes[0].text(i, v+0.01, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
axes[0].set_ylim(0, 1.1); axes[0].set_title("Retrieval Metrics")
axes[0].set_ylabel("Score"); axes[0].axhline(1.0, color="k", linewidth=0.5, linestyle="--")

axes[1].barh(range(len(EVAL)), reciprocal_ranks, color="steelblue", alpha=0.8)
axes[1].set_yticks(range(len(EVAL)))
axes[1].set_yticklabels([q[:35] for q,_ in EVAL], fontsize=7)
axes[1].set_xlabel("Reciprocal Rank"); axes[1].set_title("Per-Query Reciprocal Rank")
axes[1].axvline(mrr, color="tomato", linestyle="--", label=f"MRR={mrr:.2f}")
axes[1].legend()

plt.suptitle("Chatbot Evaluation Dashboard", fontsize=11)
plt.tight_layout(); plt.savefig("chatbot_eval.png", dpi=85, bbox_inches="tight"); plt.close()
print("\nSaved chatbot_eval.png")
