"""
Project: Zero-Shot Document Labeler
Teaches: classifying documents without any training data using semantic
         centroid similarity as a CPU-safe alternative to large NLI models.
~30 MB RAM, ~2s on CPU
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Category "anchors": 3 representative sentences per category ──────────────
CATEGORIES = {
    "Environment": [
        "climate change global warming carbon emissions greenhouse",
        "renewable energy solar wind power sustainability ecology",
        "deforestation biodiversity species extinction wildlife conservation",
    ],
    "Technology": [
        "artificial intelligence machine learning neural network algorithm",
        "software programming code computer system app",
        "robot automation drone sensor processor chip",
    ],
    "Economy": [
        "stock market investment trading profit revenue finance",
        "inflation interest rate central bank monetary policy",
        "unemployment jobs business startup company growth",
    ],
    "Health": [
        "disease treatment vaccine medicine hospital doctor",
        "nutrition diet exercise mental health wellbeing",
        "surgery cancer drug therapy clinical trial",
    ],
    "Education": [
        "school university student teacher learning curriculum",
        "exam grades scholarship research degree academic",
        "online course digital learning skill training program",
    ],
}

# ─── Test documents ───────────────────────────────────────────────────────────
TEST_DOCS = [
    ("Scientists developed solar panels that generate power at night.",   "Environment/Technology"),
    ("Central banks raised rates to combat record inflation figures.",    "Economy"),
    ("University ranked first for online engineering degree programs.",   "Education"),
    ("New mRNA vaccine reduces hospitalization rate by 85 percent.",      "Health"),
    ("Carbon capture technology removes CO2 directly from atmosphere.",   "Environment"),
    ("AI startup raised $500M to automate warehouse logistics.",          "Technology"),
    ("School dropout rate increased after the pandemic disruption.",      "Education"),
    ("Hedge funds bet against bonds as interest rate expectations rise.", "Economy"),
]

# ─── Build TF-IDF space over all anchor + test texts ─────────────────────────
all_texts = [sent for sents in CATEGORIES.values() for sent in sents]
all_texts += [doc for doc, _ in TEST_DOCS]

vec = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
vec.fit(all_texts)

cat_centroids = {}
for cat, sents in CATEGORIES.items():
    vecs = vec.transform(sents).toarray()
    cat_centroids[cat] = vecs.mean(axis=0)

print("=== Zero-Shot Document Labeler ===\n")
print(f"{'Document':<60}  {'True Label':<20}  {'Predicted'}")
print("-"*100)

correct = 0
for doc, true_label in TEST_DOCS:
    doc_vec = vec.transform([doc]).toarray()[0]
    sims    = {cat: float(np.dot(doc_vec, centroid) /
                          (np.linalg.norm(doc_vec)*np.linalg.norm(centroid)+1e-9))
               for cat, centroid in cat_centroids.items()}
    pred = max(sims, key=sims.get)
    match = "✓" if pred in true_label else "✗"
    if pred in true_label: correct += 1
    print(f"  {match} {doc[:57]:<57}  {true_label:<20}  {pred}")
    print(f"       Similarities: " + "  ".join(f"{c}={v:.3f}" for c,v in sorted(sims.items(),key=lambda x:-x[1])))

print(f"\nAccuracy (partial match): {correct}/{len(TEST_DOCS)} = {correct/len(TEST_DOCS):.1%}")
print("\nKey insight: zero-shot works by measuring semantic closeness to category anchors.")
print("No training data required — just define what each category means.")
