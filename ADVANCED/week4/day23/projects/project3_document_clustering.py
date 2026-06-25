"""
Project: Document Clustering with Word Embeddings
Teaches: averaging word vectors → document embeddings, KMeans clustering,
         PCA visualization of clustered documents.
~80 MB RAM, ~5s on CPU (model training)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ─── Tiny corpus: 24 short documents across 3 topics ─────────────────────────
DOCS = [
    # Tech (0)
    "machine learning algorithms data neural network training",
    "python programming code software development library",
    "deep learning convolutional neural network image classification",
    "natural language processing text transformer model bert",
    "computer vision object detection image recognition cnn",
    "data science pandas numpy statistics visualization analysis",
    "artificial intelligence robot automation decision tree",
    "tensorflow pytorch gradient backpropagation optimizer loss",
    # Sports (1)
    "football soccer goal player team match stadium",
    "basketball nba player score court game championship",
    "tennis serve match court racket grand slam tournament",
    "cricket bat ball wicket innings over run match",
    "swimming pool race lap freestyle backstroke medal",
    "athletics sprint marathon race track field medal",
    "cycling race tour mountain bike stage sprint",
    "boxing fight punch round champion knockout belt",
    # Food (2)
    "pasta sauce tomato cheese italian recipe dinner",
    "sushi rice fish japan raw roll wasabi ginger",
    "pizza dough topping bake oven cheese tomato basil",
    "curry spice chicken rice sauce indian dinner",
    "salad vegetable olive oil dressing fresh healthy",
    "bread flour yeast bake loaf sourdough butter",
    "chocolate cake sugar butter egg bake dessert sweet",
    "burger beef grill patty bun lettuce sauce fries",
]
TRUE_LABELS = [0]*8 + [1]*8 + [2]*8
TOPIC_NAMES = {0: "Tech", 1: "Sports", 2: "Food"}

# ─── Tokenize ─────────────────────────────────────────────────────────────────
tokenized = [doc.split() for doc in DOCS]

# ─── Train Word2Vec ───────────────────────────────────────────────────────────
print("Training Word2Vec on corpus...")
try:
    from gensim.models import Word2Vec
    w2v = Word2Vec(tokenized, vector_size=32, window=3, min_count=1, epochs=80, seed=42, workers=1)
    print(f"  Vocab size: {len(w2v.wv)}  Vector dim: {w2v.vector_size}")
    MODEL_TYPE = "word2vec"
except Exception as e:
    print(f"  gensim unavailable ({e}) — using TF-IDF fallback")
    MODEL_TYPE = "tfidf"

# ─── Document vectors ─────────────────────────────────────────────────────────
def doc_to_vec_w2v(tokens, model):
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

def doc_to_vec_tfidf(docs):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=100)
    return vec.fit_transform(docs).toarray()

if MODEL_TYPE == "word2vec":
    doc_vecs = np.array([doc_to_vec_w2v(t, w2v) for t in tokenized])
else:
    doc_vecs = doc_to_vec_tfidf(DOCS)

print(f"  Document matrix: {doc_vecs.shape}")

# ─── KMeans clustering ────────────────────────────────────────────────────────
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
pred_labels = kmeans.fit_predict(doc_vecs)

# Align predicted clusters with true labels (greedy matching)
from scipy.stats import mode
mapping = {}
for pred_c in range(3):
    mask = pred_labels == pred_c
    if mask.sum() > 0:
        true_c = mode(np.array(TRUE_LABELS)[mask], keepdims=True).mode[0]
        mapping[pred_c] = int(true_c)

aligned = np.array([mapping.get(p, p) for p in pred_labels])
accuracy = np.mean(aligned == np.array(TRUE_LABELS))
print(f"\nClustering accuracy (3 clusters): {accuracy:.1%}")

print("\nCluster assignments:")
for i, (doc, pred, true) in enumerate(zip(DOCS, pred_labels, TRUE_LABELS)):
    match = "✓" if mapping.get(pred) == true else "✗"
    print(f"  {match} [{TOPIC_NAMES[true]:6s}→cluster{pred}] {doc[:45]}")

# ─── PCA 2D visualization ─────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
pts = pca.fit_transform(doc_vecs)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors  = ["steelblue", "tomato", "limegreen"]
markers = ["o", "s", "^"]

# Left: true labels
for cls in range(3):
    mask = np.array(TRUE_LABELS) == cls
    axes[0].scatter(pts[mask, 0], pts[mask, 1],
                    c=colors[cls], marker=markers[cls], s=100,
                    label=TOPIC_NAMES[cls], edgecolors="k", linewidths=0.7)
axes[0].set_title("True Topic Labels"); axes[0].legend()
axes[0].grid(alpha=0.3); axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")

# Right: predicted clusters
for cls in range(3):
    mask = pred_labels == cls
    axes[1].scatter(pts[mask, 0], pts[mask, 1],
                    c=colors[cls], marker=markers[cls], s=100,
                    label=f"Cluster {cls}", edgecolors="k", linewidths=0.7)
axes[1].set_title(f"KMeans Clusters (acc={accuracy:.0%})")
axes[1].legend(); axes[1].grid(alpha=0.3)
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")

plt.suptitle("Document Clustering with Word Embeddings (PCA 2D)", fontsize=11)
plt.tight_layout(); plt.savefig("doc_clustering.png", dpi=85)
print("\nSaved doc_clustering.png")
plt.show()
