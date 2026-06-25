"""
Project: Multi-Topic News Classifier
Teaches: building a 6-class document classifier, confusion matrix,
         per-class metrics, error analysis.
~30 MB RAM, ~2s on CPU
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

TOPICS = {
    "Tech":    ["AI model trained on vast dataset achieves human-level performance.",
                "New programming language released for systems programming.",
                "Cybersecurity breach exposed millions of user passwords.",
                "Cloud computing costs declined due to hardware improvements.",
                "Robotics startup unveiled self-assembling construction machines.",
                "Satellite internet coverage extended to remote regions.",
                "Open-source project reached one million contributors globally.",
                "Chip shortage eased as semiconductor factories expanded production."],
    "Sports":  ["Soccer club signed a striker for record transfer fee.",
                "Gymnast won gold medal at the World Championships.",
                "Boxing champion defended title in a unanimous decision.",
                "Golf tournament ended with dramatic playoff sudden death.",
                "Cycling team dominated the mountain stages of the race.",
                "Ice hockey playoffs began with eight teams still competing.",
                "Volleyball team clinched a spot in the Olympic games.",
                "Athletics federation banned competitor for doping violations."],
    "Politics":["Parliament elected new speaker after heated debate.",
                "Climate summit agreed to reduce emissions by fifty percent.",
                "Diplomatic crisis emerged over contested territory rights.",
                "Foreign minister resigned citing policy disagreements.",
                "Trade agreement signed between five major economies.",
                "Whistleblower leaked confidential government documents.",
                "Electoral commission announced vote recount in disputed region.",
                "Opposition leader called for early general election."],
    "Health":  ["Vaccine trial shows ninety-two percent efficacy against variant.",
                "New MRI technology detects tumors at microscopic scale.",
                "Diet study links processed food to increased cancer risk.",
                "Mental health crisis among teenagers worsened post-pandemic.",
                "Gene therapy cured hereditary blindness in clinical trial.",
                "Antibiotic resistance identified as major public health threat.",
                "Hospital robotics reduced surgical recovery time by twenty percent.",
                "Alzheimer's drug showed disease progression slowdown in trial."],
    "Finance": ["Stock index hit all-time high driven by tech sector gains.",
                "Central bank held interest rates steady amid inflation concerns.",
                "Startup unicorn valued at ten billion dollars after funding round.",
                "Retail bank reported record profits from mortgage lending.",
                "Currency devaluation triggered capital outflows from emerging markets.",
                "Pension fund shifted allocation from bonds to equities.",
                "Cryptocurrency exchange collapsed due to liquidity crisis.",
                "Insurance company raised premiums citing climate-related losses."],
    "Science": ["Astronomers detected gravitational waves from merging black holes.",
                "Mars rover found evidence of ancient river channels on surface.",
                "Nuclear fusion reactor produced net energy gain for first time.",
                "Biologists sequenced genome of a newly discovered deep sea fish.",
                "Particle accelerator confirmed existence of predicted new particle.",
                "Climate scientists recorded hottest global average temperature.",
                "Paleontologists unearthed a new dinosaur species in Argentina.",
                "Chemists synthesized a new biodegradable plastic from seaweed."],
}

texts  = [t for c, docs in TOPICS.items() for t in docs]
labels = [c for c, docs in TOPICS.items() for _ in docs]

X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.25, random_state=42, stratify=labels)
vec = TfidfVectorizer(ngram_range=(1,2), max_features=2000, min_df=1)
Xtr = vec.fit_transform(X_tr); Xte = vec.transform(X_te)
lr  = LogisticRegression(max_iter=2000, C=1.5)
lr.fit(Xtr, y_tr)
y_pred = lr.predict(Xte)

print("=== Multi-Topic News Classifier ===\n")
print(classification_report(y_te, y_pred, zero_division=0))

# ─── Visualize confusion matrix ───────────────────────────────────────────────
classes = list(TOPICS.keys())
cm = confusion_matrix(y_te, y_pred, labels=classes)
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(cm, cmap="YlOrRd")
ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=40, ha="right")
ax.set_yticklabels(classes)
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, cm[i,j], ha="center", va="center", fontsize=9,
                color="white" if cm[i,j]>cm.max()/2 else "black")
plt.colorbar(im, ax=ax)
ax.set_title("6-Class Topic Classifier — Confusion Matrix")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout(); plt.savefig("topic_classifier.png", dpi=85); plt.close()
print("Saved topic_classifier.png")
