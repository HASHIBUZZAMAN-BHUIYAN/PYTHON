# NLP Reference — Text Preprocessing Template
# Covers: cleaning, tokenization, stopwords, stemming, lemmatization,
#          Bag-of-Words, TF-IDF (from scratch and sklearn).
# ~20 MB RAM, <2s on CPU

import re
import string
import numpy as np
from collections import Counter

# ─── NLTK setup ───────────────────────────────────────────────────────────────
try:
    import nltk
    nltk.download("punkt",           quiet=True)
    nltk.download("punkt_tab",       quiet=True)
    nltk.download("stopwords",       quiet=True)
    nltk.download("wordnet",         quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_OK = True
except Exception:
    NLTK_OK = False

# ─── 1. CLEANING ──────────────────────────────────────────────────────────────
def clean_text(text: str, lower=True, remove_punct=True,
               remove_numbers=False, remove_urls=True) -> str:
    if remove_urls:
        text = re.sub(r"http\S+|www\.\S+", " ", text)
    if lower:
        text = text.lower()
    if remove_numbers:
        text = re.sub(r"\d+", " ", text)
    if remove_punct:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─── 2. TOKENIZATION ──────────────────────────────────────────────────────────
def tokenize(text: str) -> list:
    if NLTK_OK:
        return word_tokenize(text)
    return text.split()

# ─── 3. STOPWORD REMOVAL ──────────────────────────────────────────────────────
ENGLISH_STOPWORDS = set()
if NLTK_OK:
    try:
        ENGLISH_STOPWORDS = set(stopwords.words("english"))
    except Exception:
        pass

def remove_stopwords(tokens: list, extra_stops=None) -> list:
    stops = ENGLISH_STOPWORDS | (set(extra_stops) if extra_stops else set())
    return [t for t in tokens if t not in stops]

# ─── 4. STEMMING & LEMMATIZATION ─────────────────────────────────────────────
_stemmer     = PorterStemmer() if NLTK_OK else None
_lemmatizer  = WordNetLemmatizer() if NLTK_OK else None

def stem(tokens: list) -> list:
    if _stemmer: return [_stemmer.stem(t) for t in tokens]
    return [t[:-3] if len(t)>5 else t for t in tokens]  # crude fallback

def lemmatize(tokens: list) -> list:
    if _lemmatizer: return [_lemmatizer.lemmatize(t) for t in tokens]
    return tokens

# ─── 5. FULL PIPELINE ────────────────────────────────────────────────────────
def preprocess(text: str, use_lemma=True, extra_stops=None) -> list:
    text   = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens, extra_stops)
    return lemmatize(tokens) if use_lemma else stem(tokens)

# ─── 6. BAG-OF-WORDS ─────────────────────────────────────────────────────────
def build_vocab(corpus: list, max_vocab=5000) -> dict:
    counter = Counter(t for doc in corpus for t in doc)
    return {w: i for i, (w, _) in enumerate(counter.most_common(max_vocab))}

def bow_vectorize(tokens: list, vocab: dict) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    for t in tokens:
        if t in vocab: vec[vocab[t]] += 1
    return vec

# ─── 7. TF-IDF FROM SCRATCH ──────────────────────────────────────────────────
def tfidf_vectorize(corpus_tokens: list, vocab: dict) -> np.ndarray:
    N = len(corpus_tokens); V = len(vocab)
    tf  = np.zeros((N, V), dtype=np.float32)
    idf = np.zeros(V, dtype=np.float32)
    for i, tokens in enumerate(corpus_tokens):
        c = Counter(tokens)
        total = max(len(tokens), 1)
        for t, freq in c.items():
            if t in vocab: tf[i, vocab[t]] = freq / total
    for j in range(V):
        df = np.sum(tf[:, j] > 0)
        idf[j] = np.log((N + 1) / (df + 1)) + 1.
    tfidf = tf * idf
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    return tfidf / np.where(norms > 0, norms, 1.)

# ─── 8. SKLEARN TF-IDF (convenience wrapper) ─────────────────────────────────
def sklearn_tfidf(corpus_texts: list, max_features=5000, ngram_range=(1,1)):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,
                          preprocessor=clean_text, tokenizer=tokenize,
                          stop_words=list(ENGLISH_STOPWORDS) or "english")
    X = vec.fit_transform(corpus_texts)
    return X, vec

# ─── 9. COSINE SIMILARITY ─────────────────────────────────────────────────────
def cosine_sim(v1, v2) -> float:
    v1, v2 = np.asarray(v1).ravel(), np.asarray(v2).ravel()
    denom  = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom > 0 else 0.

def top_k_similar(query_vec, matrix, k=5) -> list:
    """matrix: (N, D) array. Returns list of (idx, score) sorted desc."""
    sims = matrix.dot(query_vec) / (
        np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec) + 1e-9)
    top  = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in top]

# ─── DEMO ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    docs = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to interpret images.",
        "Reinforcement learning agents learn by interacting with environments.",
    ]

    print("=== Text Preprocessing Pipeline ===")
    for doc in docs[:2]:
        tokens = preprocess(doc)
        print(f"  Input : {doc[:60]}")
        print(f"  Tokens: {tokens}\n")

    print("=== TF-IDF (sklearn) ===")
    X, vectorizer = sklearn_tfidf(docs)
    print(f"  Matrix shape: {X.shape}")
    feat = vectorizer.get_feature_names_out()
    print(f"  Top features: {feat[:10].tolist()}")

    print("\n=== Cosine Similarity ===")
    query = "neural network deep learning"
    qvec  = vectorizer.transform([query]).toarray()[0]
    for i, score in top_k_similar(qvec, X.toarray(), k=3):
        print(f"  score={score:.3f}  doc: {docs[i][:55]}")
