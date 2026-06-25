"""
Project: Text Classifier Feature Importance Explorer
Teaches: extracting which words/phrases drive each class prediction,
         visualizing per-class top features as horizontal bar charts.
~25 MB RAM, ~2s on CPU
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

CORPUS = {
    "Tech":    ["AI deep learning neural network GPU chips processor.",
                "Software code programming Python algorithm data.",
                "Robotics automation sensor machine artificial intelligence.",
                "Cloud computing server database scalable infrastructure.",
                "Cybersecurity encryption firewall vulnerability patch.",
                "Quantum computing qubit processor error correction.",
                "Semiconductor chip silicon lithography fabrication.",
                "Browser JavaScript TypeScript frontend backend API."]*2,
    "Sports":  ["Football soccer goal striker midfielder defensive team.",
                "Tennis ace serve volley backhand tournament Grand Slam.",
                "Basketball dunk layup rebound assist court player.",
                "Olympic athlete sprint hurdles relay medal podium.",
                "Cricket innings wicket boundary six four over.",
                "Swimming backstroke freestyle butterfly relay pool.",
                "Golf birdie eagle putt fairway rough course club.",
                "Marathon distance runner training pace endurance race."]*2,
    "Health":  ["Cancer tumor biopsy chemotherapy radiation oncology.",
                "Vaccine immunity antibody virus pathogen infection.",
                "Surgery patient anaesthesia recovery hospital ward.",
                "Mental health therapy depression anxiety medication.",
                "Nutrition diet calories protein carbohydrate vitamin.",
                "Cardiology heart rate blood pressure cholesterol statin.",
                "Diabetes insulin glucose pancreas metabolic syndrome.",
                "Neurology brain scan MRI synapse neurotransmitter."]*2,
}

texts  = [t for c,docs in CORPUS.items() for t in docs]
labels = [c for c,docs in CORPUS.items() for _ in docs]

vec   = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
X     = vec.fit_transform(texts)
lr    = LogisticRegression(max_iter=2000, C=1.0, multi_class="ovr")
lr.fit(X, labels)
feats = vec.get_feature_names_out()

print("=== Feature Importance Explorer ===\n")
print(f"{'Class':<12}  Top 8 Predictive Words")
print("-"*70)
TOP_N = 8
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for idx, (cls, ax) in enumerate(zip(lr.classes_, axes)):
    coef = lr.coef_[idx]
    top_idx = np.argsort(coef)[-TOP_N:][::-1]
    top_words = feats[top_idx]
    top_vals  = coef[top_idx]
    print(f"  {cls:<12}: {list(top_words)}")

    colors = ["tomato" if v >= 0 else "steelblue" for v in top_vals]
    ax.barh(top_words[::-1], top_vals[::-1], color=colors[::-1], alpha=0.85,
            edgecolor="k", linewidth=0.4)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_title(f"{cls} — Top {TOP_N} Features")
    ax.set_xlabel("Coefficient")
    ax.grid(axis="x", alpha=0.3)

plt.suptitle("Per-Class Feature Importance (TF-IDF Logistic Regression)", fontsize=11)
plt.tight_layout(); plt.savefig("feature_importance.png", dpi=85, bbox_inches="tight"); plt.close()
print("\nSaved feature_importance.png")
print("\nHigh positive coefficient = strong predictor FOR that class.")
print("Features are term weights from TF-IDF + logistic regression coefficients.")
