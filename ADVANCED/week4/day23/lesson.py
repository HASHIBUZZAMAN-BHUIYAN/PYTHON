# ~200 MB RAM, ~15s on CPU
"""
Day 23: Word Embeddings
========================
Topics:
  1. Word2Vec intuition — CBOW vs Skip-gram
  2. Train a tiny Word2Vec with gensim (~500 synthetic sentences)
  3. Explore the embedding space (most_similar, similarity)
  4. Analogy arithmetic: king - man + woman ≈ queen
  5. PCA visualization of embeddings (2D scatter plot)
"""

import random
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — saves PNG instead of GUI
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# 1. WORD2VEC INTUITION
# ─────────────────────────────────────────────
print("=" * 60)
print("1. WORD2VEC INTUITION")
print("=" * 60)
print("""
Word2Vec: two architectures for learning word embeddings.

CBOW (Continuous Bag of Words):
  Context → Target
  Given surrounding words, predict the center word.
  Example: ["the", "quick", "_", "fox"] → "brown"
  - Faster to train
  - Good for frequent words

Skip-gram:
  Target → Context
  Given one word, predict surrounding words.
  Example: "brown" → ["quick", "fox"]
  - Slower to train
  - Better for rare words
  - Used in original Word2Vec paper

Both learn embeddings from co-occurrence statistics.
Words that appear in similar contexts get similar vectors.

Key hyperparameters:
  vector_size : embedding dimension (50-300 typical)
  window      : context window size (2-5 typical)
  min_count   : ignore words with fewer occurrences
  workers     : number of CPU threads
  epochs      : training passes over corpus
  sg          : 0=CBOW, 1=Skip-gram
""")


# ─────────────────────────────────────────────
# 2. BUILD SYNTHETIC CORPUS (~500 sentences)
# ─────────────────────────────────────────────
print("=" * 60)
print("2. BUILDING SYNTHETIC CORPUS")
print("=" * 60)

# Groups of semantically related words
ANIMALS   = ["cat", "dog", "lion", "tiger", "elephant", "monkey", "wolf", "bear", "fox"]
FOODS     = ["apple", "banana", "pizza", "sushi", "bread", "rice", "soup", "cake", "fish"]
PLACES    = ["city", "town", "village", "forest", "mountain", "ocean", "river", "park"]
ROYALTY   = ["king", "queen", "prince", "princess", "throne", "castle", "crown", "knight"]
GENDER    = ["man", "woman", "boy", "girl", "male", "female", "father", "mother"]
TECH      = ["computer", "internet", "software", "hardware", "algorithm", "data", "code"]
SCIENCE   = ["physics", "chemistry", "biology", "math", "experiment", "research", "lab"]
VERBS     = ["runs", "eats", "sleeps", "plays", "works", "studies", "builds", "creates"]
ADJECTIVES = ["fast", "slow", "big", "small", "smart", "strong", "brave", "wise", "royal"]

sentence_templates = [
    "the {a} and the {a} are {adj}",
    "a {a} eats {f} near the {p}",
    "the {a} runs through the {p}",
    "the {g} is {adj} and {adj}",
    "the {r} sits on the throne in the {p}",
    "the {t} {v} quickly and efficiently",
    "a {g} {v} the {t} with great skill",
    "the {s} {v} {adj} experiments in the lab",
    "{a} and {a} are natural enemies in the {p}",
    "the {r} is a {adj} {g} who rules with wisdom",
    "a {g} {v} in the {p} every day",
    "the {t} is used for {s} research",
    "the {a} is {adj} like a {r}",
    "the {g} studies {s} and {t} at school",
    "{r} and {r} rule the {p} together",
    "the {a} is {adj} and lives in the {p}",
    "a {adj} {g} eats {f} and {f} for lunch",
    "the {t} {v} like a {adj} {a}",
]

def make_sentence(template):
    return template.format(
        a   = random.choice(ANIMALS),
        f   = random.choice(FOODS),
        p   = random.choice(PLACES),
        r   = random.choice(ROYALTY),
        g   = random.choice(GENDER),
        t   = random.choice(TECH),
        s   = random.choice(SCIENCE),
        v   = random.choice(VERBS),
        adj = random.choice(ADJECTIVES),
    )

corpus_sentences = []
for _ in range(500):
    tmpl = random.choice(sentence_templates)
    sentence = make_sentence(tmpl)
    corpus_sentences.append(sentence)

tokenized_corpus = [s.lower().split() for s in corpus_sentences]

print(f"Generated {len(tokenized_corpus)} sentences")
print("Sample sentences:")
for s in corpus_sentences[:5]:
    print(f"  {s}")

vocab = set(w for sent in tokenized_corpus for w in sent)
print(f"\nVocabulary size: {len(vocab)} unique words")


# ─────────────────────────────────────────────
# 3. TRAIN WORD2VEC MODEL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. TRAINING WORD2VEC")
print("=" * 60)

model = Word2Vec(
    sentences    = tokenized_corpus,
    vector_size  = 50,       # 50-dimensional embeddings
    window       = 3,        # context window of 3 words each side
    min_count    = 2,        # ignore words appearing < 2 times
    workers      = 1,        # CPU only, single thread for reproducibility
    epochs       = 50,       # more epochs = better embeddings
    sg           = 1,        # skip-gram (better for small corpora)
    seed         = 42,
)

print(f"Vocabulary size (after min_count filter): {len(model.wv)}")
print(f"Embedding dimension: {model.wv.vector_size}")
print(f"\nSample vector for 'king' (first 10 dims):")
print(f"  {model.wv['king'][:10].round(3)}")


# ─────────────────────────────────────────────
# 4. EXPLORE EMBEDDING SPACE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. EXPLORING EMBEDDING SPACE")
print("=" * 60)

query_words = ["king", "cat", "computer", "forest", "woman"]

for word in query_words:
    if word in model.wv:
        similar = model.wv.most_similar(word, topn=5)
        print(f"Most similar to '{word}':")
        for w, score in similar:
            print(f"  {w:<15} {score:.4f}")
        print()

# Word similarity
print("Pairwise similarities:")
pairs = [
    ("king", "queen"),
    ("man", "woman"),
    ("cat", "dog"),
    ("cat", "computer"),
    ("king", "castle"),
]
for w1, w2 in pairs:
    if w1 in model.wv and w2 in model.wv:
        sim = model.wv.similarity(w1, w2)
        print(f"  similarity('{w1}', '{w2}') = {sim:.4f}")


# ─────────────────────────────────────────────
# 5. ANALOGY ARITHMETIC
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. ANALOGY ARITHMETIC: king - man + woman = ?")
print("=" * 60)

analogies = [
    ("king",  "man",    "woman",  "queen"),
    ("king",  "castle", "forest", "lion"),
    ("man",   "boy",    "girl",   "woman"),
    ("cat",   "forest", "ocean",  "fish"),
]

for pos1, neg1, pos2, expected in analogies:
    try:
        results = model.wv.most_similar(
            positive=[pos1, pos2],
            negative=[neg1],
            topn=3
        )
        top_word = results[0][0]
        top_score = results[0][1]
        status = "✓" if top_word == expected else "~"
        print(f"  {pos1} - {neg1} + {pos2} = ?")
        print(f"    Top result: '{top_word}' (score={top_score:.3f}) — expected: '{expected}' {status}")
        print(f"    Top 3: {[(w, round(s, 3)) for w, s in results]}")
    except KeyError as e:
        print(f"  Skipping (word not in vocab): {e}")
    print()

print("Note: Perfect analogies need larger corpora. Small corpus = approximate results.")


# ─────────────────────────────────────────────
# 6. PCA VISUALIZATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. PCA VISUALIZATION (saving to embeddings_pca.png)")
print("=" * 60)

viz_words = [w for w in ANIMALS + ROYALTY + GENDER + TECH[:4] + PLACES[:4]
             if w in model.wv][:35]

vectors = np.array([model.wv[w] for w in viz_words])

pca = PCA(n_components=2, random_state=42)
coords_2d = pca.fit_transform(vectors)
explained = pca.explained_variance_ratio_

print(f"PCA explained variance: {explained[0]:.1%} + {explained[1]:.1%} = {sum(explained):.1%}")

colors = []
labels_cat = []
category_map = {
    "animal":  (ANIMALS,  "steelblue"),
    "royalty": (ROYALTY,  "goldenrod"),
    "gender":  (GENDER,   "tomato"),
    "tech":    (TECH,     "mediumseagreen"),
    "place":   (PLACES,   "mediumpurple"),
}
for w in viz_words:
    assigned = False
    for cat, (word_list, color) in category_map.items():
        if w in word_list:
            colors.append(color)
            labels_cat.append(cat)
            assigned = True
            break
    if not assigned:
        colors.append("gray")
        labels_cat.append("other")

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=colors, s=100, alpha=0.7)

for i, word in enumerate(viz_words):
    ax.annotate(word, (coords_2d[i, 0], coords_2d[i, 1]),
                fontsize=9, ha='left', va='bottom',
                xytext=(4, 4), textcoords='offset points')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=cat)
                   for cat, (_, c) in category_map.items()]
ax.legend(handles=legend_elements, loc='upper right')

ax.set_title(f"Word2Vec Embeddings — PCA 2D\n(explains {sum(explained):.1%} of variance)",
             fontsize=14)
ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({explained[1]:.1%} variance)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("embeddings_pca.png", dpi=100)
plt.close()
print("Saved: embeddings_pca.png")
print("Semantically similar words should cluster together in the plot.")

print("\n--- Day 23 Complete ---")
