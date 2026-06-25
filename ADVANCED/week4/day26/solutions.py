# Advanced Day 26 — Solutions
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix

CORPUS = {
    "Tech":     ["Apple released new iPhone.", "Google trained neural networks.",
                 "Python is top ML language.", "New GPU speeds deep learning.",
                 "Microsoft invested in OpenAI.", "Quantum computing breakthrough.",
                 "Tesla autopilot uses vision.", "Linux kernel updates released."],
    "Sports":   ["Team won the championship.", "Tennis star won Grand Slam.",
                 "Sprinter broke world record.", "Basketball player scored 50.",
                 "Cricket team beat Australia.", "Swimmer dominated the event.",
                 "Marathon winner in extreme heat.", "Cycling surprise stage winner."],
    "Politics": ["Parliament approved new budget.", "President signed climate deal.",
                 "Opposition won election.", "UN debated new resolution.",
                 "Senate approved infrastructure bill.", "PM reshuffled cabinet.",
                 "EU passed data privacy law.", "Sanctions imposed after talks."],
    "Health":   ["Cancer vaccine shows results.", "Exercise reduces heart disease.",
                 "Antibiotic fights bacteria.", "Mental health apps surged.",
                 "Genome sequencing used.", "Flu vaccine reformulated.",
                 "Robot surgery performed.", "Sleep affects memory."],
    "Finance":  ["Stock market hit record high after inflation data.",
                 "Federal Reserve raised interest rates by 25 basis points.",
                 "Cryptocurrency Bitcoin surged past $60000 mark.",
                 "Bank earnings exceeded analyst expectations for Q2.",
                 "IPO raised $2 billion on the first day of trading.",
                 "Hedge fund reported 30 percent annual return.",
                 "Mortgage rates fell as bond yields declined.",
                 "Retail sales data showed consumer spending increased."],
}
texts  = [t for c,docs in CORPUS.items() for t in docs]
labels = [c for c,docs in CORPUS.items() for _ in docs]

# Ex 1
print("=== Ex 1: 5-Class with Finance ===")
vec = TfidfVectorizer(ngram_range=(1,2), max_features=1500)
X   = vec.fit_transform(texts)
lr  = LogisticRegression(max_iter=1000, C=1.0)
sc  = cross_val_score(lr, X, labels, cv=4).mean()
print(f"  5-class CV accuracy: {sc:.3f}")

# Ex 2
print("\n=== Ex 2: Confusion Matrix ===")
X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.25, random_state=42, stratify=labels)
vec2  = TfidfVectorizer(max_features=1000); Xtr=vec2.fit_transform(X_tr); Xte=vec2.transform(X_te)
lr2   = LogisticRegression(max_iter=1000).fit(Xtr, y_tr)
y_pred= lr2.predict(Xte)
classes = list(CORPUS.keys())
cm    = confusion_matrix(y_te, y_pred, labels=classes)
fig,ax= plt.subplots(figsize=(6,5))
im    = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=35, ha="right"); ax.set_yticklabels(classes)
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j,i,cm[i,j],ha="center",va="center",
                color="white" if cm[i,j]>cm.max()/2 else "black")
plt.colorbar(im,ax=ax); ax.set_title("Confusion Matrix"); ax.set_xlabel("Pred"); ax.set_ylabel("True")
plt.tight_layout(); plt.savefig("cm.png",dpi=80); plt.close(); print("  Saved cm.png")
acc=np.mean(np.array(y_pred)==np.array(y_te)); print(f"  Test accuracy: {acc:.3f}")

# Ex 3
print("\n=== Ex 3: Top Features per Class ===")
vec3=TfidfVectorizer(max_features=1500).fit(texts)
X3=vec3.transform(texts); lr3=LogisticRegression(max_iter=1000,multi_class="ovr").fit(X3,labels)
feat=vec3.get_feature_names_out()
for i,cls in enumerate(lr3.classes_):
    top5=feat[np.argsort(lr3.coef_[i])[-5:][::-1]]
    print(f"  {cls:10s}: {list(top5)}")

# Ex 4
print("\n=== Ex 4: Misclassifications ===")
for text,true,pred in zip(X_te,y_te,y_pred):
    if true!=pred: print(f"  [{true}→{pred}] {text[:60]}")

# Ex 5
print("\n=== Ex 5: UNSURE class ===")
probs=lr2.predict_proba(Xte)
max_conf=probs.max(axis=1)
threshold=0.4
unsure=np.sum(max_conf<threshold)
print(f"  Samples classified as UNSURE (<{threshold:.0%} confidence): {unsure}/{len(max_conf)}")
certain_acc=np.mean(np.array(y_pred)[max_conf>=threshold]==np.array(y_te)[max_conf>=threshold])
print(f"  Accuracy on confident predictions: {certain_acc:.3f}")
