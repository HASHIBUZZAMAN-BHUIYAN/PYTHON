# Advanced Day 06 Mini-Project — Spam Classifier
# Uses synthetic text features; no downloads needed.
# ~150 MB RAM, <20s on CPU

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

np.random.seed(42)

# ─── Synthetic spam dataset ───────────────────────────────────────────────────
spam_words  = ["win","free","prize","money","cash","click","urgent","offer","deal","buy","discount","limited"]
ham_words   = ["meeting","schedule","project","report","update","team","review","please","thanks","attached"]

def gen_message(words_pool, n_words=15):
    return " ".join(np.random.choice(words_pool, n_words, replace=True))

spam_msgs = [gen_message(spam_words + ham_words[:3], 12) for _ in range(400)]
ham_msgs  = [gen_message(ham_words  + spam_words[:2], 15) for _ in range(600)]
texts  = spam_msgs + ham_msgs
labels = [1]*400 + [0]*600

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=42)

# TF-IDF features
tfidf = TfidfVectorizer(max_features=100, ngram_range=(1,2))
X_tr_vec = tfidf.fit_transform(X_train)
X_te_vec  = tfidf.transform(X_test)

# ─── Train & compare classifiers ─────────────────────────────────────────────
models = {
    "Naive Bayes":   MultinomialNB(),
    "Random Forest": RandomForestClassifier(50, random_state=42),
    "GBM":           GradientBoostingClassifier(n_estimators=50, random_state=42),
    "LinearSVC":     LinearSVC(max_iter=2000, random_state=42),
}

results = {}
for name, m in models.items():
    m.fit(X_tr_vec, y_train)
    y_pred = m.predict(X_te_vec)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name:<20}: {acc:.4f}")

best_name = max(results, key=results.get)
print(f"\nBest: {best_name} ({results[best_name]:.4f})")
print(classification_report(y_test, models[best_name].predict(X_te_vec), target_names=["Ham","Spam"]))

# ─── Visualise ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Spam Classifier")

names = list(results.keys()); accs = list(results.values())
bars = axes[0].bar(names, accs, color="steelblue", edgecolor="white")
axes[0].bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
axes[0].set_ylim(0.8, 1.02); axes[0].set_title("Model Comparison")
axes[0].tick_params(axis="x", rotation=20)

cm = confusion_matrix(y_test, models[best_name].predict(X_te_vec))
axes[1].imshow(cm, cmap="Blues"); axes[1].set_title(f"Confusion Matrix ({best_name})")
for i in range(2):
    for j in range(2):
        axes[1].text(j,i,cm[i,j],ha="center",va="center",fontsize=14,color="white" if cm[i,j]>cm.max()/2 else "black")
axes[1].set_xticks([0,1]); axes[1].set_xticklabels(["Ham","Spam"])
axes[1].set_yticks([0,1]); axes[1].set_yticklabels(["Ham","Spam"])
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")

plt.tight_layout(); plt.savefig("spam_classifier.png", dpi=80)
print("\nSaved spam_classifier.png")
