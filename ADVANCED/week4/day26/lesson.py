# Advanced Day 26 — Text Classification & Zero-Shot with Transformers
# ~60 MB RAM, ~5s on CPU

print("""
=== Text Classification & Zero-Shot — Day 26 ===

Text Classification assigns a category label to a document.
Approaches:
  1. Bag of Words + Naive Bayes  — baseline, fast, interpretable
  2. TF-IDF + Logistic Regression — better feature weighting
  3. Zero-shot with Transformers — no labeled data needed!
""")

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# ─── 4-class dataset: Tech, Sports, Politics, Health ────────────────────────
CORPUS = {
    "Tech": [
        "Apple released new iPhone with AI features and improved battery life.",
        "Google DeepMind trained a neural network that beat humans at protein folding.",
        "Python remains the top programming language for machine learning tasks.",
        "The new GPU architecture enables faster deep learning model training.",
        "Microsoft acquired GitHub and invested billions in OpenAI research.",
        "Quantum computing breakthrough allows solving complex optimization problems.",
        "Tesla autopilot uses computer vision and sensor fusion for navigation.",
        "Linux kernel updates improve CPU scheduling and memory management.",
    ],
    "Sports": [
        "The football team won the championship after a dramatic penalty shootout.",
        "Tennis star won the Grand Slam title for the third consecutive year.",
        "Olympic sprinter broke the world record in the 100m by 0.02 seconds.",
        "Basketball player scored 50 points to lead his team to victory.",
        "Cricket team won the test match against Australia by 7 wickets.",
        "Swimming champion dominated the butterfly event at the World Championships.",
        "Marathon runner finished first despite the extreme heat conditions.",
        "The cycling race saw a surprise stage winner after a breakaway group.",
    ],
    "Politics": [
        "Parliament voted to approve the new budget with record spending.",
        "The president signed a landmark climate agreement with 50 nations.",
        "Opposition party won a surprise majority in the general election.",
        "United Nations delegates debated new peacekeeping resolution.",
        "Senate committee approved the infrastructure bill after months of talks.",
        "Prime minister announced a cabinet reshuffle following election losses.",
        "European Parliament passed new data privacy regulations for tech companies.",
        "International sanctions imposed after diplomatic relations collapsed.",
    ],
    "Health": [
        "Researchers developed a new cancer vaccine showing promising clinical trial results.",
        "Study links daily exercise to reduced risk of heart disease by 30 percent.",
        "The new antibiotic successfully targets drug-resistant bacteria infections.",
        "Mental health apps saw a surge in usage during the pandemic lockdowns.",
        "Genome sequencing helped doctors diagnose a rare genetic disorder.",
        "The flu vaccine reformulation targets three dominant strains this season.",
        "Surgeons performed a pioneering robot-assisted liver transplant operation.",
        "Sleep deprivation significantly impairs cognitive function and memory recall.",
    ],
}

texts  = [t for cat, docs in CORPUS.items() for t in docs]
labels = [cat for cat, docs in CORPUS.items() for _ in docs]
CLASS_NAMES = list(CORPUS.keys())

# ─── 1. BoW + Naive Bayes ────────────────────────────────────────────────────
print("=== 1. Bag of Words + Naive Bayes ===")
bow = CountVectorizer(max_features=500)
X   = bow.fit_transform(texts)
nb  = MultinomialNB()
sc  = cross_val_score(nb, X, labels, cv=4, scoring="accuracy")
print(f"  CV accuracy: {sc.mean():.3f} ± {sc.std():.3f}")

# ─── 2. TF-IDF + LR ──────────────────────────────────────────────────────────
print("\n=== 2. TF-IDF + Logistic Regression ===")
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=1000)
X2    = tfidf.fit_transform(texts)
lr    = LogisticRegression(max_iter=1000, C=1.0)
sc2   = cross_val_score(lr, X2, labels, cv=4, scoring="accuracy")
print(f"  CV accuracy: {sc2.mean():.3f} ± {sc2.std():.3f}")
lr.fit(X2, labels)

# ─── 3. Zero-shot classification ─────────────────────────────────────────────
print("\n=== 3. Zero-Shot Classification (no training data!) ===")
test_sentences = [
    "Scientists discovered a new method to treat Alzheimer's disease.",
    "The midfielder scored two goals in the Champions League final.",
    "Congress passed a new bill regulating social media platforms.",
    "The startup raised $200M to build autonomous warehouse robots.",
]
expected = ["Health", "Sports", "Politics", "Tech"]

def zero_shot_classify(texts, labels):
    try:
        from transformers import pipeline
        pipe = pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli",
                        device=-1)
        results = []
        for text in texts:
            r = pipe(text, candidate_labels=labels)
            results.append(r["labels"][0])
        return results
    except Exception as e:
        print(f"  HF model unavailable ({type(e).__name__}) — using TF-IDF similarity fallback")
        # Fallback: classify using centroid similarity
        from sklearn.metrics.pairwise import cosine_similarity
        cat_vecs = {}
        for cat, docs in CORPUS.items():
            cat_vecs[cat] = tfidf.transform(docs).mean(axis=0)
        preds = []
        for text in texts:
            v = tfidf.transform([text])
            sims = {cat: cosine_similarity(v, cat_vecs[cat])[0,0] for cat in CORPUS}
            preds.append(max(sims, key=sims.get))
        return preds

preds = zero_shot_classify(test_sentences, CLASS_NAMES)
print(f"\n  {'Sentence':<55}  {'Expected':>10}  {'Predicted':>10}")
print("  " + "-"*80)
for sent, exp, pred in zip(test_sentences, expected, preds):
    match = "✓" if exp==pred else "✗"
    print(f"  {match} {sent[:52]:<52}  {exp:>10}  {pred:>10}")

print("""
=== Summary ===
  BoW+NB     — fast, no context, works well for clearly separable topics
  TF-IDF+LR  — stronger weighting, bigrams help capture topic phrases
  Zero-shot  — NO training data required; useful for new categories
""")
