"""
Project: End-to-End NLP Pipeline (No External Downloads)
Teaches: chaining TF-IDF, naive Bayes, lexicon sentiment, extractive summarization,
         and simple NER into a single document-processing pipeline.
~30 MB RAM, ~2s on CPU
"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import numpy as np

# ─── 1. CORPUS ────────────────────────────────────────────────────────────────
CORPUS = [
    ("Scientists discovered a new exoplanet orbiting a distant star in the Milky Way galaxy.", "science"),
    ("The stock market surged today as tech giants reported record quarterly earnings.", "business"),
    ("The national football team won the championship after a thrilling overtime match.", "sports"),
    ("Researchers developed a vaccine that shows 95% efficacy against the new virus strain.", "science"),
    ("Central bank raised interest rates by 0.5% to combat rising inflation.", "business"),
    ("Olympic gold medalist breaks world record in the 100-meter sprint final.", "sports"),
    ("New solar panel technology achieves 40% energy conversion efficiency breakthrough.", "science"),
    ("Startup raises 50 million dollars in Series B funding for AI platform.", "business"),
    ("Basketball legend retires after 20 seasons with six championship rings.", "sports"),
    ("Climate scientists warn of accelerating ice sheet melt in Antarctica.", "science"),
    ("Merger between two major pharmaceutical companies valued at 80 billion.", "business"),
    ("Tennis star wins Grand Slam title for the fifth consecutive year.", "sports"),
]

TEXTS  = [t for t,_ in CORPUS]
LABELS = [l for _,l in CORPUS]

# ─── 2. TEXT CLASSIFICATION ──────────────────────────────────────────────────
print("=== 1. Text Classification ===")
vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1,2))
X = vectorizer.fit_transform(TEXTS)

from sklearn.model_selection import cross_val_score
nb = MultinomialNB()
lr = LogisticRegression(max_iter=200, C=1.0)
nb_cv = cross_val_score(nb, X, LABELS, cv=3, scoring="accuracy")
lr_cv = cross_val_score(lr, X, LABELS, cv=3, scoring="accuracy")
print(f"  Naive Bayes CV accuracy : {nb_cv.mean():.2f} ± {nb_cv.std():.2f}")
print(f"  Logistic Reg CV accuracy: {lr_cv.mean():.2f} ± {lr_cv.std():.2f}")

# Fit for prediction
lr.fit(X, LABELS)

# ─── 3. LEXICON SENTIMENT ────────────────────────────────────────────────────
print("\n=== 2. Sentiment Analysis ===")
POS = {"great","excellent","record","won","gold","breakthrough","wins","best","success","rises"}
NEG = {"warn","melt","combat","rising","accelerating","crisis","failed","drops","decline"}

def lexicon_sentiment(text):
    words = set(text.lower().split())
    pos   = len(words & POS); neg = len(words & NEG)
    if pos > neg:   return "POSITIVE", pos/(pos+neg+1e-9)
    if neg > pos:   return "NEGATIVE", neg/(pos+neg+1e-9)
    return "NEUTRAL", 0.5

for text in TEXTS[:6]:
    label, conf = lexicon_sentiment(text)
    print(f"  {label:8s} ({conf:.0%})  {text[:60]}")

# ─── 4. NAMED ENTITY RECOGNITION ────────────────────────────────────────────
print("\n=== 3. Named Entity Recognition ===")
NER_PATTERNS = {
    "MONEY":    r"\$[\d,]+(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:million|billion|trillion)\b",
    "PERCENT":  r"\b\d+(?:\.\d+)?%",
    "CARDINAL": r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b",
    "ORG":      r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+(?:Inc|Corp|Ltd|Bank|Company|Group)\b",
}

def simple_ner(text):
    entities = {}
    for ent_type, pattern in NER_PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches: entities[ent_type] = list(set(matches))[:3]
    return entities

for text in TEXTS[:5]:
    ents = simple_ner(text)
    print(f"  {text[:55]}")
    if ents: print(f"    Entities: {ents}")

# ─── 5. EXTRACTIVE SUMMARIZATION ─────────────────────────────────────────────
print("\n=== 4. Extractive Summarization ===")

def extractive_summary(texts, top_n=3):
    if len(texts) <= top_n: return texts
    tfidf   = TfidfVectorizer(stop_words="english")
    X_sum   = tfidf.fit_transform(texts)
    scores  = np.asarray(X_sum.sum(axis=1)).flatten()
    top_idx = scores.argsort()[::-1][:top_n]
    return [texts[i] for i in sorted(top_idx)]

summary = extractive_summary(TEXTS, top_n=3)
print("  Top-3 most informative sentences from corpus:")
for i, s in enumerate(summary, 1): print(f"    {i}. {s}")

# ─── 6. PIPELINE FUNCTION ─────────────────────────────────────────────────────
print("\n=== 5. Full Pipeline on New Documents ===")
NEW_DOCS = [
    "Tech company launches revolutionary quantum computing processor with 1000 qubits.",
    "Athletes set seven world records during the international track and field meet.",
    "Bank announces 20 billion dollar acquisition of rival financial institution.",
]
print(f"  {'Document':<55}  {'Category':<10}  Sentiment")
print("  " + "─"*80)
for doc in NEW_DOCS:
    x_new  = vectorizer.transform([doc])
    cat    = lr.predict(x_new)[0]
    sent,_ = lexicon_sentiment(doc)
    ents   = simple_ner(doc)
    print(f"  {doc[:53]:<55}  {cat:<10}  {sent}")
    if ents: print(f"  {'':55}  Entities: {ents}")
