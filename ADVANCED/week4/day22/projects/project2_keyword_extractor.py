# ~150 MB RAM, ~5s on CPU
"""
Project 2: Keyword Extractor
==============================
What it does: Extracts the top-K keywords from documents using TF-IDF scoring.
              Demonstrates how TF-IDF naturally ranks important, distinctive words.

What it teaches:
  - How TF-IDF identifies keywords (high frequency in doc, rare across corpus)
  - Using sklearn TfidfVectorizer on a small corpus
  - Ranking and displaying keywords with scores
  - Comparing keywords across different topic documents
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# ─── Sample Articles (hardcoded) ─────────────────────────────────────────────

articles = {
    "Article 1: Machine Learning": """
    Machine learning is a branch of artificial intelligence that enables computers
    to learn from data without being explicitly programmed. Algorithms improve
    through experience. Machine learning algorithms build a model based on training
    data in order to make predictions. Supervised learning, unsupervised learning,
    and reinforcement learning are three major types of machine learning.
    Neural networks, decision trees, and support vector machines are popular
    machine learning algorithms. Deep learning is a subset of machine learning
    that uses artificial neural networks with multiple layers.
    """,

    "Article 2: Climate Change": """
    Climate change refers to long-term shifts in global temperatures and weather
    patterns. Since the industrial revolution, human activities have been the
    main driver of climate change, primarily due to burning fossil fuels like
    coal, oil and gas. Climate change causes rising sea levels, extreme weather
    events, and ecosystem disruption. Greenhouse gas emissions, particularly
    carbon dioxide, trap heat in the atmosphere. Renewable energy sources like
    solar and wind power can reduce carbon emissions. International agreements
    like the Paris Agreement aim to limit global warming below 2 degrees Celsius.
    """,

    "Article 3: Blockchain Technology": """
    Blockchain is a distributed ledger technology that records transactions
    across multiple computers in a secure and transparent way. Each block
    contains a cryptographic hash of the previous block, a timestamp, and
    transaction data. Bitcoin was the first cryptocurrency to use blockchain
    technology. Smart contracts are self-executing contracts with terms written
    in code on the blockchain. Decentralized finance, or DeFi, uses blockchain
    to recreate traditional financial instruments. Ethereum is a blockchain
    platform that supports smart contracts and decentralized applications.
    """,
}

def extract_keywords(doc_dict: dict, top_k: int = 8) -> dict:
    """
    Extract top-K keywords from each document using TF-IDF.

    Args:
        doc_dict: {title: text} dictionary
        top_k: number of keywords to return per document
    Returns:
        {title: [(word, score), ...]} dictionary
    """
    titles = list(doc_dict.keys())
    texts  = list(doc_dict.values())

    # Fit TF-IDF on all documents (so IDF is computed across corpus)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),    # include bigrams for multi-word keywords
        max_features=500,
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    results = {}
    for i, title in enumerate(titles):
        scores = tfidf_matrix[i].toarray().flatten()
        top_indices = np.argsort(scores)[-top_k:][::-1]
        keywords = [(feature_names[idx], round(float(scores[idx]), 4))
                    for idx in top_indices if scores[idx] > 0]
        results[title] = keywords
    return results

def display_keywords(results: dict):
    """Pretty-print keyword extraction results."""
    for title, keywords in results.items():
        print(f"\n{title}")
        print("-" * len(title))
        for rank, (word, score) in enumerate(keywords, 1):
            bar_len = int(score * 80)
            bar = "█" * bar_len
            print(f"  {rank:2}. {word:<30} {score:.4f}  {bar}")

# ─── Run Keyword Extraction ──────────────────────────────────────────────────
print("=" * 60)
print("KEYWORD EXTRACTOR — TF-IDF Based")
print("=" * 60)

results = extract_keywords(articles, top_k=8)
display_keywords(results)

# ─── Unigrams Only (for comparison) ─────────────────────────────────────────
print("\n" + "=" * 60)
print("UNIGRAMS ONLY (no bigrams, for comparison)")
print("=" * 60)

texts = list(articles.values())
titles = list(articles.keys())

vec_uni = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
X = vec_uni.fit_transform(texts)
features = vec_uni.get_feature_names_out()

for i, title in enumerate(titles):
    scores = X[i].toarray().flatten()
    top5 = np.argsort(scores)[-5:][::-1]
    kw = [features[j] for j in top5 if scores[j] > 0]
    print(f"  {title[:30]}: {kw}")

# ─── Keyword Overlap Analysis ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEYWORD OVERLAP ANALYSIS")
print("=" * 60)

all_keywords = {}
for title, keywords in results.items():
    all_keywords[title] = set(w for w, _ in keywords)

titles_list = list(all_keywords.keys())
for i in range(len(titles_list)):
    for j in range(i + 1, len(titles_list)):
        t1, t2 = titles_list[i], titles_list[j]
        overlap = all_keywords[t1] & all_keywords[t2]
        print(f"  Overlap between '{t1[:20]}' & '{t2[:20]}':")
        print(f"    {overlap if overlap else '(no shared keywords — topics are distinct)'}")

print("\n--- Project 2 Complete ---")
