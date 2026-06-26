"""
Document Comparator
====================
What it does:
  Compares 4 built-in sample documents pairwise using two methods:
    (a) TF-IDF cosine similarity — lexical: matches shared vocabulary
    (b) Sentence-embedding cosine similarity — semantic: matches meaning
        even when different words are used

  Prints a comparison matrix for both methods, highlights the most/least
  similar pairs, and explains WHY they differ using the actual results as
  examples.

What it teaches:
  - Lexical similarity (TF-IDF): shares words -> similar. Same concept,
    different words -> dissimilar. Fast, no model needed.
  - Semantic similarity (sentence-transformers): captures meaning, so
    "automobile" and "car" score high even with zero word overlap.
  - When they diverge: a technical paper and a news article about the
    same topic may score low on TF-IDF but high on embedding similarity.
  - Practical use cases: duplicate detection (TF-IDF), topic clustering
    (embeddings), paraphrase detection (embeddings).

How to run:
  python NLP\document_comparator.py    (from PYTHON\ folder)

Estimated RAM: ~300MB (embedding model) | Time: ~2s after cache
Model: all-MiniLM-L6-v2 (sentence-transformers, ~90MB, CPU-friendly)
Falls back to TF-IDF-only if the model cannot be loaded.
No API key needed.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


# ─── SAMPLE DOCUMENTS ─────────────────────────────────────────────────────────

DOCS = {
    "A - Tech News": (
        "OpenAI has released a new version of its language model with improved "
        "reasoning capabilities and a larger context window. The model can now "
        "process longer documents and perform better on coding tasks. Researchers "
        "say the performance gains come from a new training technique."
    ),
    "B - AI Research": (
        "Recent advances in large language model pretraining have led to significant "
        "improvements in natural language understanding and code generation. "
        "Novel reinforcement learning from human feedback methods increase the "
        "alignment between model outputs and human preferences."
    ),
    "C - Sports News": (
        "The national football team secured a dramatic 2-1 victory in the final "
        "minutes of last night's championship match. The striker scored a "
        "stunning long-range goal that sent fans into wild celebration. "
        "The coach praised the team's resilience and tactical discipline."
    ),
    "D - Food Review": (
        "The new Italian restaurant downtown offers an exceptional dining experience. "
        "The pasta dishes are made fresh daily using traditional recipes from "
        "southern Italy. The wine selection is excellent and the service is warm "
        "and attentive. Highly recommended for a special occasion."
    ),
}

DOC_LABELS = list(DOCS.keys())
DOC_TEXTS  = [DOCS[k] for k in DOC_LABELS]


# ─── TF-IDF SIMILARITY ────────────────────────────────────────────────────────

def tfidf_similarity_matrix(texts: list[str]) -> np.ndarray:
    """Compute pairwise TF-IDF cosine similarity matrix."""
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_mat  = vectorizer.fit_transform(texts)
    return sklearn_cosine(tfidf_mat).tolist()


# ─── EMBEDDING SIMILARITY ─────────────────────────────────────────────────────

def embedding_similarity_matrix(texts: list[str]):
    """
    Compute pairwise cosine similarity using sentence embeddings.
    Returns (matrix, None) on success, or (None, error_msg) on failure.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model  = SentenceTransformer("all-MiniLM-L6-v2")
        vecs   = model.encode(texts, show_progress_bar=False)
        norms  = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        matrix = (norms @ norms.T).tolist()
        return matrix, None
    except Exception as e:
        return None, str(e)


# ─── DISPLAY HELPERS ──────────────────────────────────────────────────────────

def print_matrix(labels: list[str], matrix, title: str):
    short = [lbl.split("-")[0].strip() for lbl in labels]  # "A", "B", etc.
    n     = len(labels)
    col_w = 8

    print(f"\n  {title}")
    header = "  " + " " * 4 + "".join(f"{s:>{col_w}}" for s in short)
    print(header)
    print("  " + "-" * (4 + col_w * n))
    for i in range(n):
        row = f"  {short[i]:<4}" + "".join(
            f"{matrix[i][j]:>{col_w}.3f}" for j in range(n)
        )
        print(row)


def find_extreme_pairs(labels: list[str], matrix, exclude_self=True):
    n = len(labels)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((matrix[i][j], labels[i], labels[j]))
    pairs.sort()
    return pairs[0], pairs[-1]  # least similar, most similar


# ─── DEMO ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 68)
    print("  DOCUMENT COMPARATOR DEMO")
    print("=" * 68)

    print("\n  Documents:")
    for label, text in DOCS.items():
        print(f"  {label}: \"{text[:70]}...\"")

    # ── TF-IDF ────────────────────────────────────────────────────────────────
    tfidf_mat = tfidf_similarity_matrix(DOC_TEXTS)
    print_matrix(DOC_LABELS, tfidf_mat, "(A) TF-IDF Cosine Similarity (lexical)")
    tfidf_low, tfidf_high = find_extreme_pairs(DOC_LABELS, tfidf_mat)
    print(f"\n  Most similar  (TF-IDF): {tfidf_high[1]} <-> {tfidf_high[2]}  score={tfidf_high[0]:.3f}")
    print(f"  Least similar (TF-IDF): {tfidf_low[1]} <-> {tfidf_low[2]}  score={tfidf_low[0]:.3f}")

    # ── EMBEDDING ─────────────────────────────────────────────────────────────
    print("\n  Loading embedding model (all-MiniLM-L6-v2)...")
    emb_mat, err = embedding_similarity_matrix(DOC_TEXTS)
    if emb_mat is not None:
        print_matrix(DOC_LABELS, emb_mat, "(B) Embedding Cosine Similarity (semantic)")
        emb_low, emb_high = find_extreme_pairs(DOC_LABELS, emb_mat)
        print(f"\n  Most similar  (Embed): {emb_high[1]} <-> {emb_high[2]}  score={emb_high[0]:.3f}")
        print(f"  Least similar (Embed): {emb_low[1]} <-> {emb_low[2]}  score={emb_low[0]:.3f}")

        # ── Side-by-side comparison for the A-B pair ──────────────────────────
        idx_a = DOC_LABELS.index("A - Tech News")
        idx_b = DOC_LABELS.index("B - AI Research")
        tf_ab  = tfidf_mat[idx_a][idx_b]
        emb_ab = emb_mat[idx_a][idx_b]

        print()
        print("  ANALYSIS: Docs A (Tech News) vs B (AI Research)")
        print("  Both are about AI/language models but use different vocabulary.")
        print(f"  TF-IDF score  : {tf_ab:.3f}  (shares some words: 'model', 'language')")
        print(f"  Embedding score: {emb_ab:.3f}  (captures shared MEANING even when words differ)")
        print()
        print("  KEY INSIGHT:")
        print("  TF-IDF only sees word overlap. If doc A says 'automobile' and")
        print("  doc B says 'car', TF-IDF scores them as unrelated (0.0).")
        print("  Embeddings map both to similar vectors -> high similarity score.")
        print("  Use TF-IDF for fast duplicate detection (exact phrasing matters).")
        print("  Use embeddings for topic clustering (meaning matters, not words).")
    else:
        print(f"  Embedding model not available ({err}).")
        print("  Showing TF-IDF analysis only.")
        print()
        print("  KEY INSIGHT (TF-IDF only):")
        print("  Docs A and B (both AI-topic) score higher than A vs C (AI vs sports).")
        print("  But TF-IDF misses when same concept uses different words.")

    print()
    print("[DONE] document_comparator.py complete")
