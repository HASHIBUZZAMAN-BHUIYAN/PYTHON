# Advanced Day 24 — Sentiment Analysis
# ~60 MB RAM, ~5s on CPU

import numpy as np

print("""
=== Sentiment Analysis — Day 24 ===

Three approaches:
  1. Lexicon-based  — count positive/negative words (fast, no training)
  2. ML (TF-IDF+LR) — train a classifier on labeled examples
  3. Transformer    — use a pretrained model via HuggingFace pipeline
""")

# ─── SYNTHETIC DATASET ────────────────────────────────────────────────────────
import nltk; nltk.download("punkt", quiet=True); nltk.download("punkt_tab", quiet=True)

POS_REVIEWS = [
    "This movie was absolutely wonderful, I loved every minute!",
    "Fantastic product, works perfectly and arrived quickly.",
    "Amazing performance, the best I have ever seen.",
    "Great value for money, highly recommend this to everyone.",
    "Excellent quality, very happy with my purchase.",
    "Loved it! Will definitely buy again.",
    "Outstanding service and beautiful product.",
    "The best experience I've had in a long time.",
    "Simply brilliant, exceeded all my expectations.",
    "A masterpiece. Cannot recommend highly enough.",
]
NEG_REVIEWS = [
    "Terrible product, broke after one day of use.",
    "Awful experience, customer service was useless.",
    "Worst movie I have ever seen, complete waste of time.",
    "Poor quality, very disappointed with this purchase.",
    "Horrible. The item arrived damaged and looks nothing like the picture.",
    "Do not buy this. It stopped working within a week.",
    "Dreadful. Cheap materials and terrible build quality.",
    "Disgusting service, they refused to refund my money.",
    "Complete garbage. Save your money and avoid this.",
    "Extremely disappointed. This is the worst product ever.",
]
REVIEWS = POS_REVIEWS + NEG_REVIEWS
LABELS  = [1]*10 + [0]*10   # 1=positive, 0=negative

# ─── 1. LEXICON-BASED ────────────────────────────────────────────────────────
print("=== 1. Lexicon-Based Sentiment ===")
POS_WORDS = {"wonderful","loved","fantastic","amazing","great","excellent","brilliant",
             "outstanding","best","recommend","happy","perfect","masterpiece","loved"}
NEG_WORDS = {"terrible","awful","worst","poor","horrible","bad","dreadful","disgusting",
             "garbage","disappointed","useless","damaged","broke","avoid","waste"}

def lexicon_sentiment(text):
    words = set(text.lower().split())
    pos = len(words & POS_WORDS); neg = len(words & NEG_WORDS)
    if pos > neg:   return 1, "POSITIVE"
    if neg > pos:   return 0, "NEGATIVE"
    return -1, "NEUTRAL"

correct = sum(lexicon_sentiment(r)[0]==l for r,l in zip(REVIEWS,LABELS) if lexicon_sentiment(r)[0]!=-1)
total   = sum(1 for r in REVIEWS if lexicon_sentiment(r)[0]!=-1)
print(f"  Lexicon accuracy: {correct}/{total} = {correct/total:.1%}")

# ─── 2. TF-IDF + LOGISTIC REGRESSION ─────────────────────────────────────────
print("\n=== 2. TF-IDF + Logistic Regression ===")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

vec = TfidfVectorizer(ngram_range=(1,2), max_features=500)
X   = vec.fit_transform(REVIEWS)
y   = np.array(LABELS)
lr  = LogisticRegression(max_iter=1000, C=1.0)
cv_scores = cross_val_score(lr, X, y, cv=4, scoring="accuracy")
print(f"  CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
lr.fit(X, y)   # fit on full set for demo

# Show top features
feat   = vec.get_feature_names_out()
coefs  = lr.coef_[0]
top_pos = feat[np.argsort(coefs)[-5:]][::-1]
top_neg = feat[np.argsort(coefs)[:5]]
print(f"  Top positive features: {list(top_pos)}")
print(f"  Top negative features: {list(top_neg)}")

# ─── 3. HUGGINGFACE TRANSFORMER (with fallback) ───────────────────────────────
print("\n=== 3. HuggingFace Transformer Pipeline ===")
import time

def hf_sentiment(texts):
    try:
        from transformers import pipeline
        pipe = pipeline("sentiment-analysis",
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        device=-1)
        return [("POSITIVE" if r["label"]=="POSITIVE" else "NEGATIVE") for r in pipe(texts)]
    except Exception as e:
        print(f"  HF unavailable ({type(e).__name__}) — using lexicon fallback")
        return [lexicon_sentiment(t)[1] for t in texts]

test_texts = ["This is absolutely incredible!", "Total waste of money, awful."]
t0 = time.time()
hf_preds = hf_sentiment(test_texts)
elapsed  = time.time() - t0
for text, pred in zip(test_texts, hf_preds):
    print(f"  [{pred:8s}] {text}")
print(f"  Inference time: {elapsed:.2f}s for {len(test_texts)} texts")

# ─── 4. COMPARISON SUMMARY ───────────────────────────────────────────────────
print("""
=== Summary: When to use each approach ===
  Lexicon  — no training, fast, works zero-shot; lower accuracy on subtle text
  TF-IDF+LR— needs labeled data, interpretable, very fast; misses context
  Transformer — highest accuracy, context-aware; ~260MB model, slower

Rule of thumb:
  • Have labels + need speed/interpretability → TF-IDF+LR
  • No labels / new domain → start with lexicon
  • Need best accuracy + RAM available → Transformer
""")
