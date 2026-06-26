"""
Topic Extractor (LDA + TF-IDF Keyword Clustering)
===================================================
What it does:
  Discovers hidden topics in a small corpus of 8 built-in sample documents
  covering 4 distinct topics (technology, healthcare, sports, finance).

  Two approaches are run side-by-side:
    (a) LDA (Latent Dirichlet Allocation) via sklearn — probabilistic topic
        model that discovers latent word-topic distributions. Each document
        gets a probability distribution over topics.
    (b) TF-IDF keyword clustering — scores words by TF-IDF within each doc,
        clusters docs by their top-keyword overlap (simpler, more transparent).

  Output shows:
    - Top keywords for each discovered topic
    - Which document belongs to which topic
    - Comparison of the two approaches on this small corpus

What it teaches:
  - LDA: the statistical intuition (doc = mix of topics, topic = mix of words)
  - Why LDA needs more data to work well (8 docs is tiny — shows limitations)
  - TF-IDF keyword approach: simpler, more interpretable on small datasets
  - When to use which: LDA for large corpora (100+ docs), keyword approach
    for small corpora or when interpretability matters

How to run:
  python NLP\topic_extractor.py    (from PYTHON\ folder)

Estimated RAM: ~100MB | Time: <2s
Model note: LDA via sklearn.decomposition.LatentDirichletAllocation
  (100% offline, no download). TF-IDF also from sklearn.
No API key needed.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import defaultdict


# ─── SAMPLE CORPUS (8 documents, 4 topics, 2 docs each) ─────────────────────

CORPUS = [
    ("Tech-1",
     "Artificial intelligence and machine learning are transforming software "
     "development. Neural networks can now recognise images and translate languages "
     "with high accuracy. Cloud computing enables scalable AI model training."),

    ("Tech-2",
     "The latest smartphone features a powerful processor and advanced camera system "
     "with computational photography. Battery technology has improved significantly. "
     "App developers are integrating AI features into mobile software."),

    ("Health-1",
     "Researchers have developed a new drug that targets cancer cells without harming "
     "healthy tissue. Clinical trials showed significant tumour reduction in patients. "
     "The treatment uses monoclonal antibodies to block tumour growth pathways."),

    ("Health-2",
     "Regular exercise reduces the risk of heart disease and type 2 diabetes. "
     "A balanced diet rich in vegetables and whole grains supports immune function. "
     "Mental health is closely linked to physical activity and sleep quality."),

    ("Sports-1",
     "The football team won the championship after a penalty shootout. "
     "The striker scored the winning goal in the final minute of extra time. "
     "Fans celebrated throughout the night after the historic victory."),

    ("Sports-2",
     "The tennis player won her third Grand Slam title with a commanding performance. "
     "Her serve speed reached 180 km/h during the final set. "
     "The coach praised her mental toughness and physical conditioning."),

    ("Finance-1",
     "Stock markets fell sharply as inflation concerns drove investors to bonds. "
     "The central bank raised interest rates to curb rising consumer prices. "
     "Analysts warned of a potential recession in the coming quarters."),

    ("Finance-2",
     "Cryptocurrency prices surged after major institutional investors entered the market. "
     "Bitcoin reached a new all-time high as demand for digital assets increased. "
     "Regulators are considering new frameworks to govern decentralised finance."),
]

DOC_IDS    = [c[0] for c in CORPUS]
DOC_TEXTS  = [c[1] for c in CORPUS]
N_TOPICS   = 4


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def top_n_words(feature_names: list, component: np.ndarray, n: int = 6) -> list[str]:
    """Return top-n words for a topic component (LDA row or TF-IDF centroid)."""
    idx = np.argsort(component)[::-1][:n]
    return [feature_names[i] for i in idx]


# ─── (A) LDA TOPIC MODELLING ─────────────────────────────────────────────────

def run_lda(texts: list[str], n_topics: int):
    # Count vectorizer (LDA needs raw counts, not TF-IDF)
    count_vec = CountVectorizer(stop_words="english", min_df=1, max_df=0.9)
    X_counts  = count_vec.fit_transform(texts)
    features  = count_vec.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42, max_iter=30,
        learning_method="batch",
    )
    doc_topic = lda.fit_transform(X_counts)   # shape: (n_docs, n_topics)

    # Top keywords per topic
    topic_words = [top_n_words(features, comp, n=6) for comp in lda.components_]

    # Dominant topic per document
    doc_assignments = np.argmax(doc_topic, axis=1)

    return topic_words, doc_assignments, doc_topic


# ─── (B) TF-IDF KEYWORD CLUSTERING ───────────────────────────────────────────

def run_tfidf_keywords(texts: list[str], n_topics: int):
    """
    Simpler approach:
    1. Compute TF-IDF matrix
    2. For each doc, extract top keywords
    3. Cluster docs by keyword overlap using a greedy grouping
    """
    tfidf_vec = TfidfVectorizer(stop_words="english", max_features=200)
    X_tfidf   = tfidf_vec.fit_transform(texts)
    features  = tfidf_vec.get_feature_names_out()

    # Top-3 keywords per doc
    doc_top_kws = []
    for i in range(X_tfidf.shape[0]):
        row   = X_tfidf[i].toarray().flatten()
        idx   = np.argsort(row)[::-1][:3]
        doc_top_kws.append([features[j] for j in idx])

    # Assign clusters: docs with any shared top keyword go to same cluster
    assignments  = [-1] * len(texts)
    cluster_id   = 0
    kw_to_cluster = {}
    for i, kws in enumerate(doc_top_kws):
        found = None
        for kw in kws:
            if kw in kw_to_cluster:
                found = kw_to_cluster[kw]
                break
        if found is None:
            found = cluster_id
            cluster_id += 1
        assignments[i] = found
        for kw in kws:
            kw_to_cluster.setdefault(kw, found)

    # Collect top words per cluster
    cluster_words = defaultdict(list)
    for i, cid in enumerate(assignments):
        cluster_words[cid].extend(doc_top_kws[i])

    topic_words = []
    seen_clusters = sorted(set(assignments))
    for cid in seen_clusters:
        # Deduplicate and take top-6 by frequency
        freq = defaultdict(int)
        for w in cluster_words[cid]:
            freq[w] += 1
        top = sorted(freq, key=freq.get, reverse=True)[:6]
        topic_words.append(top)

    return topic_words, assignments, seen_clusters


# ─── DEMO ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 68)
    print("  TOPIC EXTRACTOR DEMO")
    print("=" * 68)
    print(f"  Corpus: {len(CORPUS)} documents, expecting ~{N_TOPICS} topics")
    print()

    # ── LDA ───────────────────────────────────────────────────────────────────
    print("  (A) LDA (Latent Dirichlet Allocation)")
    print("  " + "-" * 60)
    lda_words, lda_assign, doc_topic_probs = run_lda(DOC_TEXTS, N_TOPICS)

    for t_idx, words in enumerate(lda_words):
        print(f"  Topic {t_idx}: {', '.join(words)}")
    print()

    print("  Document assignments (dominant topic):")
    for doc_id, t_idx in zip(DOC_IDS, lda_assign):
        probs_str = " ".join(f"T{i}:{doc_topic_probs[list(DOC_IDS).index(doc_id)][i]:.2f}"
                             for i in range(N_TOPICS))
        kw_label  = ", ".join(lda_words[t_idx][:3])
        print(f"    {doc_id:<12} -> Topic {t_idx}  [{probs_str}]  ({kw_label})")

    # ── TF-IDF ────────────────────────────────────────────────────────────────
    print()
    print("  (B) TF-IDF Keyword Clustering")
    print("  " + "-" * 60)
    kw_words, kw_assign, kw_clusters = run_tfidf_keywords(DOC_TEXTS, N_TOPICS)

    cluster_labels = {cid: ", ".join(kw_words[i][:4])
                      for i, cid in enumerate(kw_clusters)}
    for t_idx, words in zip(kw_clusters, kw_words):
        print(f"  Cluster {t_idx}: {', '.join(words)}")
    print()

    print("  Document assignments:")
    for doc_id, cid in zip(DOC_IDS, kw_assign):
        print(f"    {doc_id:<12} -> Cluster {cid}  ({cluster_labels.get(cid, '?')})")

    # ── Comparison ────────────────────────────────────────────────────────────
    print()
    print("  COMPARISON:")
    print("  LDA: probabilistic - each doc gets a mix of topics.")
    print("       Needs 100+ docs to find stable topics. With 8 docs,")
    print("       results may shift each run or merge unrelated topics.")
    print("  TF-IDF clustering: deterministic, interpretable, good on small")
    print("       corpora. Trade-off: misses semantic relations across docs.")
    print("  Rule of thumb: use LDA for 100+ docs, keyword approach for <20.")
    print()
    print("[DONE] topic_extractor.py complete")
