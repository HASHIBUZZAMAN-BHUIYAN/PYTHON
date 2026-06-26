"""
Project: POS Tag Analyzer — Writing Style Profiler
Teaches: using POS distribution to measure writing style, comparing
         fiction vs technical text, visualization.
~30 MB RAM, ~2s on CPU
"""
import re
from collections import Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FICTION = """
The old lighthouse keeper watched the stormy sea with tired, worried eyes.
Dark clouds gathered menacingly above the rocky shore while howling winds
shook the ancient tower violently. She carefully climbed the narrow, creaking
stairs and lit the brilliant beacon, hoping desperately that lost sailors would
safely find their way home through the terrible, dark night.
"""

TECHNICAL = """
The algorithm processes input data using a recursive function that iterates
over each element. The system initializes variables and allocates memory
before executing the main computation loop. Results are stored in a
structured array and subsequently written to the output file.
The program terminates after releasing all allocated resources.
"""

ADJECTIVES  = {"old","tired","worried","dark","stormy","rocky","ancient","narrow",
               "creaking","brilliant","terrible","lost","safe","lost",
               "recursive","structured","main","allocated","simple","complex"}
ADVERBS     = {"carefully","menacingly","violently","desperately","safely","subsequently",
               "before","after","over","each"}
VERBS_PAST  = {"watched","gathered","shook","climbed","lit","stored","executed",
               "wrote","released","processes","initializes","allocates","terminates","iterates"}

def simple_pos_profile(text):
    words = re.findall(r"[a-z]+", text.lower())
    pos_counts = {"NOUN":0, "VERB":0, "ADJ":0, "ADV":0, "OTHER":0}
    for w in words:
        if w in ADJECTIVES:   pos_counts["ADJ"]  += 1
        elif w in ADVERBS:    pos_counts["ADV"]  += 1
        elif w in VERBS_PAST: pos_counts["VERB"] += 1
        elif len(w) > 4 and not w.endswith(("ing","ly","ed","er")):
            pos_counts["NOUN"] += 1
        else:
            pos_counts["OTHER"] += 1
    total = max(sum(pos_counts.values()), 1)
    return {k: v/total for k, v in pos_counts.items()}

# Try NLTK for better tagging
def nltk_pos_profile(text):
    try:
        import nltk
        for p in ["punkt","punkt_tab","averaged_perceptron_tagger","averaged_perceptron_tagger_eng"]:
            nltk.download(p, quiet=True)
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        SIMPLIFY = {
            "NN":"NOUN","NNS":"NOUN","NNP":"NOUN","NNPS":"NOUN",
            "VB":"VERB","VBD":"VERB","VBG":"VERB","VBN":"VERB","VBP":"VERB","VBZ":"VERB",
            "JJ":"ADJ","JJR":"ADJ","JJS":"ADJ",
            "RB":"ADV","RBR":"ADV","RBS":"ADV",
        }
        simple = [SIMPLIFY.get(t, "OTHER") for _, t in pos_tags]
        total  = max(len(simple), 1)
        counts = Counter(simple)
        return {k: counts[k]/total for k in ["NOUN","VERB","ADJ","ADV","OTHER"]}
    except Exception:
        return simple_pos_profile(text)

print("=== POS Style Profiler ===\n")
fiction_prof  = nltk_pos_profile(FICTION)
technical_prof= nltk_pos_profile(TECHNICAL)

cats = ["NOUN","VERB","ADJ","ADV","OTHER"]
print(f"{'Category':<10} {'Fiction':>10}  {'Technical':>10}")
print("-"*35)
for c in cats:
    fi = fiction_prof.get(c, 0)
    te = technical_prof.get(c, 0)
    print(f"{c:<10} {fi:>10.3f}  {te:>10.3f}")

print("\nObservation: Fiction uses more adjectives and adverbs (descriptive).")
print("Technical text uses more nouns (precise terminology).\n")

# ─── Visualize ────────────────────────────────────────────────────────────────
x = range(len(cats))
w = 0.35
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar([i-w/2 for i in x], [fiction_prof.get(c,0) for c in cats],
       width=w, label="Fiction", color="steelblue", alpha=0.85, edgecolor="k", linewidth=0.5)
ax.bar([i+w/2 for i in x], [technical_prof.get(c,0) for c in cats],
       width=w, label="Technical", color="tomato", alpha=0.85, edgecolor="k", linewidth=0.5)
ax.set_xticks(list(x)); ax.set_xticklabels(cats)
ax.set_ylabel("Fraction of Words"); ax.set_title("Writing Style POS Profile")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("pos_style.png", dpi=85); plt.close()
print("Saved pos_style.png")
