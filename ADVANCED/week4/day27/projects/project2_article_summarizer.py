"""
Project: Automatic Article Summarizer
Teaches: extractive summarization with TF-IDF + cosine centroid method,
         compression ratio, side-by-side comparison.
~20 MB RAM, ~1s on CPU
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ARTICLES = {
    "Electric Vehicles": """
    The global electric vehicle market has experienced unprecedented growth over the past decade.
    Sales of battery electric vehicles surpassed 10 million units worldwide in 2022 alone.
    China leads the world in EV adoption with over 6 million vehicles sold in that year.
    Government subsidies and improving battery technology have driven the rapid price decline.
    The average cost of a lithium-ion battery pack fell from $1200 per kWh in 2010 to just $132 in 2022.
    Major automakers including Ford, General Motors and Volkswagen have committed to fully electric lineups.
    Charging infrastructure remains a key challenge with only 1.8 million public chargers globally.
    Range anxiety affects many potential buyers despite average ranges exceeding 300 miles per charge.
    Battery recycling programs are being developed to address environmental concerns at end of life.
    Analysts predict EVs will account for 60 percent of all new car sales by 2030 globally.
    """,
    "Quantum Computing": """
    Quantum computers exploit quantum mechanical phenomena to perform calculations beyond classical limits.
    Unlike classical bits that hold 0 or 1, quantum bits or qubits can exist in superposition of both states.
    IBM unveiled a 433-qubit quantum processor called Osprey in November 2022.
    Google claimed quantum supremacy in 2019 when its Sycamore chip solved a problem in 200 seconds.
    The same calculation would take classical supercomputers approximately 10,000 years to complete.
    Error correction remains the biggest obstacle to practical quantum computing applications.
    Current quantum computers require cooling to temperatures near absolute zero using liquid helium.
    Potential applications include drug discovery, cryptography breaking, and materials science.
    Post-quantum cryptography standards are being developed to protect against future quantum attacks.
    Experts estimate fault-tolerant quantum computers capable of real applications are still a decade away.
    """,
}

def extract_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.strip()) > 15]

def tfidf_summarize(sentences, n=3):
    if len(sentences) <= n: return list(range(len(sentences)))
    vec = TfidfVectorizer(stop_words="english")
    X   = vec.fit_transform(sentences)
    scores = X.sum(axis=1).A1
    return sorted(np.argsort(scores)[-n:])

def centroid_summarize(sentences, n=3):
    if len(sentences) <= n: return list(range(len(sentences)))
    vec      = TfidfVectorizer(stop_words="english")
    X        = vec.fit_transform(sentences)
    centroid = X.mean(axis=0)
    sims     = cosine_similarity(X, centroid).flatten()
    return sorted(np.argsort(sims)[-n:])

print("=== Article Summarizer ===\n")
for title, article in ARTICLES.items():
    sents   = extract_sentences(article)
    orig_w  = len(article.split())

    idx_tfidf    = tfidf_summarize(sents, n=3)
    idx_centroid = centroid_summarize(sents, n=3)

    summ_tfidf    = " ".join(sents[i] for i in idx_tfidf)
    summ_centroid = " ".join(sents[i] for i in idx_centroid)

    print(f"{'='*60}")
    print(f"ARTICLE: {title} ({orig_w} words, {len(sents)} sentences)\n")
    print(f"[TF-IDF Extractive — {len(summ_tfidf.split())} words]:")
    print(f"  {summ_tfidf}\n")
    print(f"[Centroid Similarity — {len(summ_centroid.split())} words]:")
    print(f"  {summ_centroid}\n")

    ratio_tfidf    = len(summ_tfidf.split()) / orig_w
    ratio_centroid = len(summ_centroid.split()) / orig_w
    print(f"  Compression: TF-IDF={ratio_tfidf:.1%}  Centroid={ratio_centroid:.1%}")
    overlap = set(idx_tfidf) & set(idx_centroid)
    print(f"  Methods agree on sentences: {sorted(overlap)} out of {min(len(idx_tfidf),len(idx_centroid))}\n")
