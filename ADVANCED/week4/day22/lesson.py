# ~150 MB RAM, ~10s on CPU
"""
Day 22: Text Preprocessing
===========================
Topics:
  1. Tokenization (split vs NLTK word_tokenize)
  2. Stopwords removal
  3. Stemming (PorterStemmer) vs Lemmatization (WordNetLemmatizer)
  4. Regex cleaning
  5. Bag-of-Words from scratch
  6. TF-IDF from scratch AND with sklearn
"""

import re
import math
from collections import Counter, defaultdict

import nltk
nltk.download('punkt',      quiet=True)
nltk.download('punkt_tab',  quiet=True)
nltk.download('stopwords',  quiet=True)
nltk.download('wordnet',    quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────
# 1. TOKENIZATION
# ─────────────────────────────────────────────
print("=" * 60)
print("1. TOKENIZATION")
print("=" * 60)

text = "Hello! I'm learning NLP in Python 3.11 — it's amazing, isn't it?"

# Simple split (naive)
tokens_split = text.split()
print("split()  :", tokens_split[:8], "...")

# NLTK word_tokenize (handles contractions, punctuation)
tokens_nltk = word_tokenize(text)
print("word_tokenize:", tokens_nltk)

# Key difference: split keeps "it?" as one token; word_tokenize separates "it" "?"
print("\nwhy word_tokenize is better:")
print("  split keeps 'isn't' as one token:", "isn't" in tokens_split)
print("  word_tokenize splits 'isn't' ->", [t for t in tokens_nltk if "n" in t.lower() and "'" in t])


# ─────────────────────────────────────────────
# 2. STOPWORDS REMOVAL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. STOPWORDS REMOVAL")
print("=" * 60)

STOP_WORDS = set(stopwords.words('english'))
print(f"Total English stop words: {len(STOP_WORDS)}")
print("Examples:", list(STOP_WORDS)[:10])

sample = "The quick brown fox jumps over the lazy dog near the river"
tokens = word_tokenize(sample.lower())
filtered = [t for t in tokens if t.isalpha() and t not in STOP_WORDS]
print(f"\nOriginal : {tokens}")
print(f"Filtered : {filtered}")
print(f"Reduction: {len(tokens)} → {len(filtered)} tokens")


# ─────────────────────────────────────────────
# 3. STEMMING vs LEMMATIZATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. STEMMING vs LEMMATIZATION")
print("=" * 60)

stemmer  = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running", "ran", "runs", "easily", "fairly", "studies",
         "studying", "studied", "better", "goods", "caring"]

print(f"{'Word':<12} {'Stem':<12} {'Lemma':<12}")
print("-" * 36)
for w in words:
    stem  = stemmer.lemmatize(w)          # PorterStemmer.stem()
    stem  = stemmer.stem(w)
    lemma = lemmatizer.lemmatize(w, pos='v')  # verb form
    print(f"{w:<12} {stem:<12} {lemma:<12}")

print("""
Key differences:
  Stemming  : fast, rule-based, may produce non-words ("studi")
  Lemmatization: slower, dictionary-based, always real words
  Lemmatization needs POS (part-of-speech) for accuracy:
    lemmatize('better', pos='a') → 'good'
    lemmatize('running', pos='v') → 'run'
""")

lemma_adj = lemmatizer.lemmatize('better', pos='a')
lemma_v   = lemmatizer.lemmatize('running', pos='v')
print(f"  'better' (adj) → {lemma_adj}")
print(f"  'running' (v)  → {lemma_v}")


# ─────────────────────────────────────────────
# 4. REGEX CLEANING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. REGEX CLEANING")
print("=" * 60)

def clean_text(text: str) -> str:
    """Standard text cleaning pipeline."""
    text = text.lower()                              # lowercase
    text = re.sub(r'https?://\S+', '', text)         # remove URLs
    text = re.sub(r'@\w+', '', text)                 # remove @mentions
    text = re.sub(r'#\w+', '', text)                 # remove #hashtags
    text = re.sub(r'[^a-z\s]', '', text)             # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()         # collapse whitespace
    return text

raw_tweets = [
    "Check out https://example.com #NLP is AMAZING!! @user123",
    "Can't believe it's 2024... NLP just keeps getting better!!!",
    "Visit http://bit.ly/abc123 for more info @johndoe #machinelearning",
]

print(f"{'Raw':<52} {'Cleaned'}")
print("-" * 80)
for t in raw_tweets:
    print(f"{t[:50]:<52} {clean_text(t)}")


# ─────────────────────────────────────────────
# 5. BAG-OF-WORDS FROM SCRATCH
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. BAG-OF-WORDS FROM SCRATCH")
print("=" * 60)

corpus = [
    "I love natural language processing",
    "language processing is fun and exciting",
    "I love machine learning and NLP",
    "natural language understanding is key",
]

def build_bow(corpus):
    """Build Bag-of-Words vocabulary and document-term matrix."""
    # Step 1: tokenize and build vocabulary
    tokenized = [doc.lower().split() for doc in corpus]
    vocab = sorted(set(word for doc in tokenized for word in doc))
    word2idx = {w: i for i, w in enumerate(vocab)}

    # Step 2: build document-term matrix
    dtm = []
    for tokens in tokenized:
        counts = Counter(tokens)
        row = [counts.get(w, 0) for w in vocab]
        dtm.append(row)
    return vocab, dtm

vocab, dtm = build_bow(corpus)
print(f"Vocabulary ({len(vocab)} words): {vocab}")
print(f"\nDocument-Term Matrix (rows=docs, cols=words):")
header = f"{'Doc':<6}" + "".join(f"{w[:6]:<7}" for w in vocab[:8]) + "..."
print(header)
for i, row in enumerate(dtm):
    print(f"Doc{i:<3}" + "".join(f"{v:<7}" for v in row[:8]) + "...")


# ─────────────────────────────────────────────
# 6. TF-IDF FROM SCRATCH
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. TF-IDF FROM SCRATCH")
print("=" * 60)

def compute_tfidf(corpus):
    """
    TF-IDF from scratch.
    TF(t,d)  = count(t in d) / total_words(d)
    IDF(t)   = log( (N+1) / (df(t)+1) ) + 1   [sklearn smooth variant]
    TFIDF    = TF * IDF
    """
    tokenized = [doc.lower().split() for doc in corpus]
    N = len(tokenized)

    # Build vocabulary
    vocab = sorted(set(w for doc in tokenized for w in doc))
    word2idx = {w: i for i, w in enumerate(vocab)}

    # Document frequency
    df = defaultdict(int)
    for tokens in tokenized:
        for w in set(tokens):
            df[w] += 1

    # IDF (smoothed)
    idf = {w: math.log((N + 1) / (df[w] + 1)) + 1 for w in vocab}

    # TF-IDF matrix
    tfidf_matrix = []
    for tokens in tokenized:
        total = len(tokens)
        tf = Counter(tokens)
        row = [(tf[w] / total) * idf[w] for w in vocab]
        tfidf_matrix.append(row)

    return vocab, tfidf_matrix, idf

vocab_tfidf, tfidf_mat, idf = compute_tfidf(corpus)

print("Top TF-IDF scores per document:")
for i, row in enumerate(tfidf_mat):
    top = sorted(zip(vocab_tfidf, row), key=lambda x: -x[1])[:4]
    print(f"  Doc {i}: {[(w, round(s, 3)) for w, s in top]}")

print("\nIDF scores (lower = more common):")
sorted_idf = sorted(idf.items(), key=lambda x: x[1])
print("  Most common words:", [(w, round(v, 3)) for w, v in sorted_idf[:4]])
print("  Rarest words:    ", [(w, round(v, 3)) for w, v in sorted_idf[-4:]])


# ─────────────────────────────────────────────
# 7. TF-IDF WITH SKLEARN
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. TF-IDF WITH SKLEARN (CountVectorizer & TfidfVectorizer)")
print("=" * 60)

# CountVectorizer = Bag of Words
cv = CountVectorizer()
X_count = cv.fit_transform(corpus)
print("CountVectorizer feature names:", cv.get_feature_names_out()[:10])
print("Shape:", X_count.shape)

# TfidfVectorizer = TF-IDF
tv = TfidfVectorizer()
X_tfidf = tv.fit_transform(corpus)
print("\nTfidfVectorizer feature names:", tv.get_feature_names_out()[:10])
print("Shape:", X_tfidf.shape)

# TF-IDF with n-grams
tv_bigram = TfidfVectorizer(ngram_range=(1, 2), max_features=20)
X_bigram = tv_bigram.fit_transform(corpus)
print("\nBigram TF-IDF features:", tv_bigram.get_feature_names_out())

# Quick classifier demo
train_texts = [
    "I love this product it's great",
    "Wonderful experience, highly recommend",
    "Excellent quality, very happy",
    "This is terrible, worst ever",
    "Awful experience, do not buy",
    "Completely useless, wasted money",
]
train_labels = [1, 1, 1, 0, 0, 0]

test_texts = ["Amazing product, love it", "Complete garbage, never again"]
test_labels = [1, 0]

tv2 = TfidfVectorizer()
X_train = tv2.fit_transform(train_texts)
X_test  = tv2.transform(test_texts)

clf = LogisticRegression()
clf.fit(X_train, train_labels)
preds = clf.predict(X_test)
print(f"\nQuick classifier demo:")
for txt, pred, true in zip(test_texts, preds, test_labels):
    label = "POSITIVE" if pred == 1 else "NEGATIVE"
    print(f"  '{txt[:40]}' → {label} (true={'POS' if true==1 else 'NEG'})")

print("\n--- Day 22 Complete ---")
