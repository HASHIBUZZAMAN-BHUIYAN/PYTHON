# ~150 MB RAM, ~10s on CPU
"""
Day 22: Text Preprocessing — Solutions
========================================
Complete working solutions to all 5 exercises.
"""

import re
import math
from collections import Counter, defaultdict
import nltk
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# ─────────────────────────────────────────────
# Solution 1: Full Preprocessing Pipeline
# ─────────────────────────────────────────────
print("=" * 60)
print("SOLUTION 1: Full Preprocessing Pipeline")
print("=" * 60)

def preprocess(text: str) -> list:
    """Full NLP preprocessing pipeline."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and t.isalpha()]
    tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens]
    return tokens

text1 = "She was running quickly through the beautiful https://example.com forest!"
text2 = "The cats are eating fish while dogs are barking loudly"

print(f"Input  : {text1}")
print(f"Output : {preprocess(text1)}")
print()
print(f"Input  : {text2}")
print(f"Output : {preprocess(text2)}")


# ─────────────────────────────────────────────
# Solution 2: Stemming vs Lemmatization Comparison
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SOLUTION 2: Stemming vs Lemmatization Comparison")
print("=" * 60)

stemmer    = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["geese", "mice", "running", "better", "studies", "wolves", "went", "happily"]

print(f"{'Word':<12} {'Stem':<12} {'Lemma(noun)':<14} {'Lemma(verb)':<12}")
print("-" * 50)
for w in words:
    stem       = stemmer.stem(w)
    lemma_noun = lemmatizer.lemmatize(w, pos='n')
    lemma_verb = lemmatizer.lemmatize(w, pos='v')
    print(f"{w:<12} {stem:<12} {lemma_noun:<14} {lemma_verb:<12}")

print("""
Observations:
  'geese' (noun) → goose  (lemmatizer handles irregular plurals)
  'went'  (verb) → go     (lemmatizer handles irregular past tense)
  Stemmer often cuts too aggressively; lemmatizer produces real words.
""")


# ─────────────────────────────────────────────
# Solution 3: Custom TF-IDF Vectorizer
# ─────────────────────────────────────────────
print("=" * 60)
print("SOLUTION 3: Custom TF-IDF Vectorizer")
print("=" * 60)

class SimpleTfidfVectorizer:
    """TF-IDF vectorizer built from scratch."""

    def __init__(self):
        self.vocab_ = []
        self.idf_   = {}

    def fit_transform(self, corpus: list) -> list:
        tokenized = [doc.lower().split() for doc in corpus]
        N = len(tokenized)

        all_words = sorted(set(w for doc in tokenized for w in doc))
        self.vocab_ = all_words

        df = defaultdict(int)
        for tokens in tokenized:
            for w in set(tokens):
                df[w] += 1

        # IDF (smoothed, sklearn-style)
        self.idf_ = {w: math.log((N + 1) / (df[w] + 1)) + 1 for w in all_words}

        matrix = []
        for tokens in tokenized:
            total = len(tokens)
            tf = Counter(tokens)
            row = [(tf[w] / total) * self.idf_[w] for w in all_words]
            matrix.append(row)
        return matrix

    def get_feature_names(self) -> list:
        return self.vocab_

docs = [
    "python is great for data science",
    "machine learning with python",
    "data science and machine learning",
    "python programming is fun",
]

vec = SimpleTfidfVectorizer()
matrix = vec.fit_transform(docs)
features = vec.get_feature_names()

print("Top 3 TF-IDF words per document:")
for i, row in enumerate(matrix):
    top3 = sorted(zip(features, row), key=lambda x: -x[1])[:3]
    print(f"  Doc {i} ('{docs[i][:30]}...'): {[(w, round(s, 3)) for w, s in top3]}")


# ─────────────────────────────────────────────
# Solution 4: Bag-of-Words Cosine Similarity
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SOLUTION 4: Bag-of-Words Cosine Similarity")
print("=" * 60)

def cosine_similarity(vec1: list, vec2: list) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a ** 2 for a in vec1))
    norm2 = math.sqrt(sum(b ** 2 for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

documents = [
    "I love pizza and pasta",
    "Pizza and pasta are my favorite foods",
    "Machine learning is fascinating",
    "Deep learning and neural networks are powerful",
]

tv = TfidfVectorizer()
X = tv.fit_transform(documents).toarray()

labels = [f"Doc{i}" for i in range(len(documents))]
print(f"{'':8}", end="")
for l in labels:
    print(f"{l:>8}", end="")
print()

for i in range(len(documents)):
    print(f"{labels[i]:<8}", end="")
    for j in range(len(documents)):
        sim = cosine_similarity(X[i].tolist(), X[j].tolist())
        print(f"{sim:>8.3f}", end="")
    print()

print("\nAnalysis: Doc0 and Doc1 (food) are more similar to each other.")
print("          Doc2 and Doc3 (ML) are more similar to each other.")


# ─────────────────────────────────────────────
# Solution 5: N-gram Language Model
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SOLUTION 5: Bigram Language Model")
print("=" * 60)

def build_bigram_model(corpus: list) -> dict:
    """
    Build a bigram frequency model.
    Returns: {word: Counter({next_word: count})}
    """
    model = defaultdict(Counter)
    for sentence in corpus:
        tokens = ['<START>'] + sentence.lower().split() + ['<END>']
        for w1, w2 in zip(tokens[:-1], tokens[1:]):
            model[w1][w2] += 1
    return dict(model)

def predict_next(model: dict, word: str, top_n: int = 3) -> list:
    """Return top_n most likely next words for given word."""
    if word not in model:
        return []
    return model[word].most_common(top_n)

corpus = [
    "the cat sat on the mat",
    "the cat ate the fish",
    "the dog sat on the floor",
    "the dog ate the bone",
    "a cat and a dog are friends",
]

model = build_bigram_model(corpus)

print("Bigram model predictions:")
for query in ["the", "cat", "dog", "<START>", "ate"]:
    preds = predict_next(model, query, top_n=3)
    print(f"  After '{query}': {preds}")

print("\nNote: '<START>' gives words that begin sentences.")

print("\n--- Day 22 Solutions Complete ---")
