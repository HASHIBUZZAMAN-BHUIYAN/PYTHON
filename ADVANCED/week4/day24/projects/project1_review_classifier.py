"""
Project: Movie/Product Review Sentiment Classifier
Teaches: TF-IDF feature extraction, Logistic Regression classification,
         evaluation with confusion matrix and per-class metrics.
~25 MB RAM, ~2s on CPU
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ─── Dataset: 300 synthetic reviews ──────────────────────────────────────────
np.random.seed(42)
POS_TEMPLATES = [
    "This {} is absolutely wonderful and I love it.",
    "Fantastic {}, works perfectly and exceeded my expectations.",
    "Amazing quality {}, highly recommend to everyone.",
    "Excellent {} with great value for the price.",
    "Outstanding {} that I will definitely buy again.",
    "Brilliant {}, the best I have used in a long time.",
    "Superb {}, very happy with my purchase.",
    "Incredible {}, it does exactly what it promises.",
    "Wonderful {}, perfect gift and great packaging.",
    "Lovely {}, simple to use and very effective.",
]
NEG_TEMPLATES = [
    "This {} is terrible and broke after one day.",
    "Awful {}, complete waste of money, do not buy.",
    "Poor quality {}, very disappointed with this purchase.",
    "Horrible {}, arrived damaged and customer service ignored me.",
    "Worst {} I have ever bought, total garbage.",
    "Dreadful {}, nothing like the description online.",
    "Disgusting {}, the smell alone made me return it immediately.",
    "Pathetic {}, stopped working within a week.",
    "Dismal {}, cheap materials and terrible construction.",
    "Appalling {}, the company refused to refund me.",
]
PRODUCTS = ["product","item","device","gadget","tool","purchase","order","package"]

reviews, labels = [], []
for _ in range(150):
    t = POS_TEMPLATES[np.random.randint(10)].format(np.random.choice(PRODUCTS))
    reviews.append(t); labels.append(1)
for _ in range(150):
    t = NEG_TEMPLATES[np.random.randint(10)].format(np.random.choice(PRODUCTS))
    reviews.append(t); labels.append(0)

# ─── Train / test split ───────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(reviews, labels, test_size=0.25, random_state=42)

# ─── TF-IDF + LR ─────────────────────────────────────────────────────────────
vec = TfidfVectorizer(ngram_range=(1,2), max_features=1000, min_df=1)
Xtr = vec.fit_transform(X_tr)
Xte = vec.transform(X_te)
lr  = LogisticRegression(max_iter=1000, C=2.0)
lr.fit(Xtr, y_tr)

y_pred = lr.predict(Xte)
y_prob = lr.predict_proba(Xte)[:,1]

print("=== Review Sentiment Classifier ===\n")
print(classification_report(y_te, y_pred, target_names=["Negative","Positive"]))

# ─── Confusion matrix ─────────────────────────────────────────────────────────
cm = confusion_matrix(y_te, y_pred)
print("Confusion matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

# ─── Sample predictions ───────────────────────────────────────────────────────
print("\n5 Sample Predictions:")
for i in range(5):
    label  = "POSITIVE" if y_te[i]==1 else "NEGATIVE"
    pred   = "POSITIVE" if y_pred[i]==1 else "NEGATIVE"
    conf   = max(y_prob[i], 1-y_prob[i])
    match  = "✓" if y_te[i]==y_pred[i] else "✗"
    print(f"  {match} [{label:8s}→{pred:8s}] conf={conf:.2f}  {X_te[i][:55]}")

# ─── Visualize ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].imshow(cm, cmap="Blues")
axes[0].set_xticks([0,1]); axes[0].set_yticks([0,1])
axes[0].set_xticklabels(["Negative","Positive"]); axes[0].set_yticklabels(["Negative","Positive"])
for i in range(2):
    for j in range(2):
        axes[0].text(j,i,cm[i,j],ha="center",va="center",fontsize=14,
                     color="white" if cm[i,j]>cm.max()/2 else "black")
axes[0].set_title("Confusion Matrix"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

axes[1].hist(y_prob[np.array(y_te)==1], bins=15, alpha=0.7, label="Positive", color="green")
axes[1].hist(y_prob[np.array(y_te)==0], bins=15, alpha=0.7, label="Negative", color="red")
axes[1].axvline(0.5, color="k", linestyle="--"); axes[1].set_title("Confidence Distributions")
axes[1].set_xlabel("P(Positive)"); axes[1].legend()

plt.suptitle("Review Sentiment Classifier", fontsize=11)
plt.tight_layout(); plt.savefig("review_classifier.png", dpi=85); plt.close()
print("\nSaved review_classifier.png")
