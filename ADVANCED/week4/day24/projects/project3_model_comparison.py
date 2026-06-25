"""
Project: Sentiment Model Comparison Dashboard
Teaches: benchmarking multiple sentiment approaches on the same dataset,
         speed vs accuracy tradeoff bar charts.
~30 MB RAM, ~3s on CPU
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

# ─── 60-sample dataset ───────────────────────────────────────────────────────
POS = [
    "Absolutely wonderful, exceeded all my expectations.",
    "Fantastic experience, will definitely return.",
    "Great quality and amazing value for money.",
    "Brilliant product, highly recommend to everyone.",
    "Excellent service, very satisfied with my purchase.",
    "Outstanding performance, the best I have tried.",
    "Superb quality, perfectly happy with this item.",
    "Incredible value, love every aspect of it.",
    "Wonderful gift, arrived on time and beautifully packaged.",
    "Amazing, simply the best product in its category.",
] * 3  # 30 positive
NEG = [
    "Terrible product, total waste of money.",
    "Awful quality, very disappointed with this purchase.",
    "Poor service, would never buy again.",
    "Horrible experience, arrived damaged and broken.",
    "Worst product I have ever bought, complete garbage.",
    "Dreadful quality, nothing like the advertised picture.",
    "Disgusting customer service, they refused to help.",
    "Absolutely useless, stopped working after one day.",
    "Terrible, do not waste your money on this.",
    "Appalling quality, fell apart immediately.",
] * 3  # 30 negative
TEXTS  = POS + NEG
LABELS = np.array([1]*30 + [0]*30)

# ─── Lexicon approach ─────────────────────────────────────────────────────────
POS_WORDS = {"wonderful","fantastic","great","brilliant","excellent","outstanding","superb",
             "incredible","wonderful","amazing","love","best","happy","recommend","satisfied"}
NEG_WORDS = {"terrible","awful","poor","horrible","worst","dreadful","disgusting","useless",
             "appalling","garbage","waste","disappointed","damaged","broken","bad"}

def lexicon_acc(texts, labels):
    preds=[]
    for t in texts:
        w=set(t.lower().split()); p=len(w&POS_WORDS); n=len(w&NEG_WORDS)
        preds.append(1 if p>=n else 0)
    return np.mean(np.array(preds)==labels)

# ─── Benchmark all models ─────────────────────────────────────────────────────
results = {}

# Lexicon
t0 = time.time()
acc = lexicon_acc(TEXTS, LABELS)
results["Lexicon"] = {"acc": acc, "time": time.time()-t0}

# BoW + Naive Bayes
for name, (vec, clf) in {
    "BoW+NB":    (CountVectorizer(max_features=500), MultinomialNB()),
    "TF-IDF+LR": (TfidfVectorizer(max_features=500,ngram_range=(1,2)), LogisticRegression(max_iter=1000)),
    "TF-IDF+SVM":(TfidfVectorizer(max_features=500), LinearSVC(max_iter=2000)),
}.items():
    t0 = time.time()
    X = vec.fit_transform(TEXTS)
    sc= cross_val_score(clf, X, LABELS, cv=4, scoring="accuracy").mean()
    results[name] = {"acc": sc, "time": time.time()-t0}

print("=== Sentiment Model Comparison ===\n")
print(f"{'Model':<15} {'Accuracy':>10}  {'Time (s)':>10}")
print("-"*40)
for model, r in sorted(results.items(), key=lambda x: -x[1]["acc"]):
    print(f"{model:<15} {r['acc']:>10.3f}  {r['time']:>10.4f}")

# ─── Visualize ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
models  = list(results.keys())
accs    = [results[m]["acc"] for m in models]
times   = [results[m]["time"] for m in models]
colors  = ["#4CAF50","#2196F3","#FF9800","#9C27B0"]

axes[0].bar(models, accs, color=colors, alpha=0.85, edgecolor="k", linewidth=0.7)
axes[0].axhline(0.5, color="red", linestyle="--", linewidth=1)
axes[0].set_ylim(0, 1.05); axes[0].set_title("Accuracy Comparison")
axes[0].set_ylabel("Cross-Val Accuracy")
for i, v in enumerate(accs):
    axes[0].text(i, v+0.01, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

axes[1].bar(models, times, color=colors, alpha=0.85, edgecolor="k", linewidth=0.7)
axes[1].set_title("Speed Comparison (seconds)")
axes[1].set_ylabel("Execution Time (s)")
for i, v in enumerate(times):
    axes[1].text(i, v+0.001, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

plt.suptitle("Sentiment Model Comparison: Accuracy vs Speed", fontsize=11)
plt.tight_layout(); plt.savefig("model_comparison.png", dpi=85); plt.close()
print("\nSaved model_comparison.png")
print("\nConclusion: TF-IDF+SVM usually has highest accuracy; Lexicon is fastest but least accurate.")
