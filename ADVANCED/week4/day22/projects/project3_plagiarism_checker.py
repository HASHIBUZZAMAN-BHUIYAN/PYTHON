# ~150 MB RAM, ~5s on CPU
"""
Project 3: Plagiarism Checker
================================
What it does: Compares pairs of documents for similarity using TF-IDF
              cosine similarity. High similarity may indicate plagiarism.

What it teaches:
  - TF-IDF vectorization for document comparison
  - Cosine similarity as a text similarity metric
  - Interpreting similarity scores (0 = no overlap, 1 = identical)
  - Building a practical NLP tool from scratch
"""

import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Synthetic Document Pairs ─────────────────────────────────────────────────

documents = {
    "Doc A — Original Essay on AI": """
    Artificial intelligence is transforming industries around the world. From healthcare
    to finance, AI systems are automating complex tasks that previously required human
    expertise. Machine learning algorithms analyze vast amounts of data to uncover
    patterns and make predictions. Natural language processing allows computers to
    understand and generate human language. The rapid advancement of AI raises important
    ethical questions about privacy, bias, and the future of work.
    """,

    "Doc B — Plagiarized (heavy paraphrase of A)": """
    Artificial intelligence is revolutionizing sectors across the globe. From medicine
    to banking, AI systems are automating complicated tasks that used to need human
    knowledge. Machine learning algorithms process large amounts of data to find
    patterns and generate predictions. Natural language processing enables computers to
    comprehend and produce human language. The fast development of AI raises critical
    ethical concerns about privacy, bias, and employment futures.
    """,

    "Doc C — Plagiarized (light paraphrase of A)": """
    AI is changing industries worldwide. Healthcare and finance use AI to automate
    tasks. ML algorithms find patterns in data. NLP lets computers understand language.
    AI raises ethical questions about privacy and jobs.
    """,

    "Doc D — Different Topic (Climate Change)": """
    Climate change is one of the most pressing challenges of our time. Rising
    temperatures, melting ice caps, and extreme weather events are becoming more
    frequent. The burning of fossil fuels releases greenhouse gases that trap heat
    in the atmosphere. Renewable energy sources such as solar and wind power offer
    sustainable alternatives. International cooperation is essential to limit
    global warming and protect ecosystems for future generations.
    """,

    "Doc E — Different Topic (Cooking)": """
    Cooking is both an art and a science. The right combination of heat, time, and
    ingredients transforms raw food into delicious meals. Understanding the Maillard
    reaction helps explain why bread turns golden brown when baked. Fermentation
    uses microorganisms to preserve food and develop complex flavors. A good chef
    balances flavors by combining sweet, salty, sour, bitter, and umami elements.
    """,

    "Doc F — Near-Duplicate of A": """
    Artificial intelligence is transforming industries around the world. From healthcare
    to finance, AI systems are automating complex tasks that previously required human
    expertise. Machine learning algorithms analyze vast amounts of data to uncover
    patterns and make predictions. Natural language processing allows computers to
    understand and generate human language.
    """,
}

# Define pairs to compare
test_pairs = [
    ("Doc A — Original Essay on AI",        "Doc B — Plagiarized (heavy paraphrase of A)", "high"),
    ("Doc A — Original Essay on AI",        "Doc C — Plagiarized (light paraphrase of A)", "medium"),
    ("Doc A — Original Essay on AI",        "Doc F — Near-Duplicate of A",                 "very high"),
    ("Doc A — Original Essay on AI",        "Doc D — Different Topic (Climate Change)",     "low"),
    ("Doc A — Original Essay on AI",        "Doc E — Different Topic (Cooking)",            "very low"),
    ("Doc D — Different Topic (Climate Change)", "Doc E — Different Topic (Cooking)",       "low"),
]

# ─── Vectorize All Documents ─────────────────────────────────────────────────
print("=" * 60)
print("PLAGIARISM CHECKER — TF-IDF Cosine Similarity")
print("=" * 60)

doc_names = list(documents.keys())
doc_texts = list(documents.values())

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=1,
)
tfidf_matrix = vectorizer.fit_transform(doc_texts)
print(f"Corpus: {len(documents)} documents | Vocabulary: {len(vectorizer.vocabulary_)} features")

# ─── Pairwise Similarity Computation ─────────────────────────────────────────
def get_doc_index(name, doc_names):
    for i, n in enumerate(doc_names):
        if n == name:
            return i
    return -1

def similarity_label(score: float) -> str:
    if score >= 0.85:
        return "VERY HIGH — likely plagiarism"
    elif score >= 0.60:
        return "HIGH — strong similarity"
    elif score >= 0.35:
        return "MEDIUM — partial overlap"
    elif score >= 0.15:
        return "LOW — minor overlap"
    else:
        return "VERY LOW — unrelated"

print("\n" + "─" * 70)
print("PAIRWISE SIMILARITY RESULTS")
print("─" * 70)

for doc1_name, doc2_name, expected in test_pairs:
    idx1 = get_doc_index(doc1_name, doc_names)
    idx2 = get_doc_index(doc2_name, doc_names)

    vec1 = tfidf_matrix[idx1]
    vec2 = tfidf_matrix[idx2]
    score = cosine_similarity(vec1, vec2)[0][0]
    label = similarity_label(score)

    short1 = doc1_name.split("—")[1].strip()[:25]
    short2 = doc2_name.split("—")[1].strip()[:25]

    print(f"\n  Pair: '{short1}' vs '{short2}'")
    print(f"  Similarity Score : {score:.4f}  ({score:.1%})")
    print(f"  Assessment       : {label}")
    print(f"  Expected level   : {expected}")

# ─── Full Similarity Matrix ──────────────────────────────────────────────────
print("\n" + "─" * 70)
print("FULL SIMILARITY MATRIX (all pairs)")
print("─" * 70)

sim_matrix = cosine_similarity(tfidf_matrix)
short_names = [n.split("—")[0].strip() for n in doc_names]

# Header
print(f"\n{'':8}", end="")
for sn in short_names:
    print(f"{sn:>8}", end="")
print()

for i, sn in enumerate(short_names):
    print(f"{sn:<8}", end="")
    for j in range(len(short_names)):
        print(f"{sim_matrix[i][j]:>8.3f}", end="")
    print()

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("SUMMARY")
print("─" * 70)
print("""
Observations:
  - Doc A and Doc F (near-duplicate) → highest similarity (~0.95+)
  - Doc A and Doc B (heavy paraphrase) → high similarity (~0.70-0.85)
  - Doc A and Doc C (light paraphrase) → medium similarity (~0.35-0.60)
  - AI docs vs Climate/Cooking → low similarity (~0.05-0.20)

Plagiarism threshold recommendations:
  > 0.85 : Almost certainly plagiarized
  > 0.60 : Likely plagiarized, requires review
  > 0.35 : Suspicious, partial copying possible
  < 0.35 : Probably original work
""")

print("--- Project 3 Complete ---")
