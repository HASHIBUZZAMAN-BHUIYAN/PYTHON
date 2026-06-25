# NLP Reference — Embedding Similarity Template
# Word2Vec training (gensim), sentence embeddings (sentence-transformers),
# cosine similarity search, document clustering.
# ~150 MB RAM, <10s on CPU (model loading)

import numpy as np

# ─── 1. WORD2VEC WITH GENSIM ─────────────────────────────────────────────────
def train_word2vec(sentences: list, vector_size=50, window=3,
                   min_count=1, epochs=20, workers=1):
    """
    Train a Word2Vec model on a list of tokenized sentences.
    sentences: list of list of strings
    Returns: trained gensim Word2Vec model
    """
    from gensim.models import Word2Vec
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        seed=42,
    )
    return model

def find_similar(model, word: str, topn=5) -> list:
    """Return top-N similar words. Returns [] if word not in vocab."""
    try:
        return model.wv.most_similar(word, topn=topn)
    except KeyError:
        return []

def word_analogy(model, pos1, neg1, pos2, topn=3) -> list:
    """A is to B as C is to ?: e.g. analogy(model,'king','man','woman')"""
    try:
        return model.wv.most_similar(positive=[pos1, pos2], negative=[neg1], topn=topn)
    except KeyError as e:
        return []

def doc_to_vec(tokens: list, model) -> np.ndarray:
    """Average word vectors for a list of tokens → document embedding."""
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if not vecs: return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)

# ─── 2. SENTENCE TRANSFORMERS (tiny model) ───────────────────────────────────
def load_sentence_encoder(model_name="all-MiniLM-L6-v2"):
    """
    Load a small sentence encoder (~90 MB).
    Falls back to TF-IDF if sentence-transformers not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return model, "sentence_transformer"
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        model = TfidfVectorizer(max_features=512)
        return model, "tfidf_fallback"

def encode_sentences(sentences: list, model, model_type="sentence_transformer") -> np.ndarray:
    if model_type == "sentence_transformer":
        return model.encode(sentences, show_progress_bar=False)
    else:
        from scipy.sparse import issparse
        X = model.fit_transform(sentences)
        return X.toarray() if issparse(X) else np.array(X)

# ─── 3. COSINE SIMILARITY SEARCH ─────────────────────────────────────────────
def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A: (m,d), B: (n,d) → returns (m,n) similarity matrix."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A_norm @ B_norm.T

def semantic_search(query: str, corpus: list, embeddings: np.ndarray,
                    encoder, model_type: str, k=3) -> list:
    """
    Find top-k most relevant sentences from corpus for a query.
    Returns: list of (sentence, score) tuples.
    """
    q_emb = encode_sentences([query], encoder, model_type)
    sims  = cosine_sim_matrix(q_emb, embeddings)[0]
    top_k = np.argsort(sims)[::-1][:k]
    return [(corpus[i], float(sims[i])) for i in top_k]

# ─── 4. CLUSTERING ────────────────────────────────────────────────────────────
def cluster_documents(embeddings: np.ndarray, n_clusters=3, random_state=42):
    """KMeans clustering on document embeddings."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels, km

def visualize_embeddings_2d(embeddings: np.ndarray, labels=None,
                             texts=None, title="Embeddings", save_path=None):
    """Project embeddings to 2D with PCA and plot."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    pca  = PCA(n_components=2, random_state=42)
    pts  = pca.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(pts[:,0], pts[:,1],
                         c=labels if labels is not None else "steelblue",
                         cmap="tab10", alpha=0.8, s=80)
    if texts:
        for i, txt in enumerate(texts):
            ax.annotate(txt[:20], (pts[i,0], pts[i,1]), fontsize=7, alpha=0.7)
    if labels is not None:
        plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title(title); ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=80); plt.close()
    else: plt.show()

# ─── DEMO ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Sample corpus
    sentences_raw = [
        ["machine", "learning", "artificial", "intelligence"],
        ["deep", "learning", "neural", "networks", "layers"],
        ["python", "programming", "language", "code"],
        ["natural", "language", "processing", "text"],
        ["computer", "vision", "image", "recognition"],
        ["robot", "arm", "control", "motion"],
        ["database", "sql", "query", "data"],
    ]
    texts = [" ".join(s) for s in sentences_raw]

    print("=== Word2Vec ===")
    w2v = train_word2vec(sentences_raw, vector_size=16, epochs=50)
    print(f"  Vocab size: {len(w2v.wv)}")
    similar = find_similar(w2v, "learning", topn=3)
    print(f"  Words similar to 'learning': {similar}")

    print("\n=== Sentence Encoder ===")
    encoder, etype = load_sentence_encoder()
    embs = encode_sentences(texts, encoder, etype)
    print(f"  Embedding shape: {embs.shape}  (type: {etype})")

    print("\n=== Semantic Search ===")
    results = semantic_search("neural networks for image tasks", texts, embs, encoder, etype, k=3)
    for txt, score in results:
        print(f"  score={score:.3f}  {txt}")

    print("\n=== Clustering ===")
    labels, _ = cluster_documents(embs, n_clusters=3)
    for i, (txt, lbl) in enumerate(zip(texts, labels)):
        print(f"  [{lbl}] {txt}")
