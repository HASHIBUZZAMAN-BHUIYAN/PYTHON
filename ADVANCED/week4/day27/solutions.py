# Advanced Day 27 — Solutions
import re, random
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CORPUS = """
The quick brown fox jumps over the lazy dog in the meadow.
A fox is clever and hunts at night under the dark sky.
The meadow is full of flowers and tall green grass near the stream.
Brown bears hunt for fish in the cold mountain stream every morning.
Dense forests are home to many wild birds and animals.
Birds sing beautifully when the warm sun rises over the green hills.
Winter brings cold winds and heavy snow to the mountain valleys each year.
The old lighthouse keeper watches the stormy sea every dark night.
Scientists discovered that deep sleep improves memory and learning greatly.
Electric vehicles reduce emissions and save fuel costs over time.
"""
tokens = re.sub(r"[^a-z\s]", "", CORPUS.lower()).split()

# Ex 1
print("=== Ex 1: N-gram comparison ===")
def build_ngram(tokens, n):
    model = defaultdict(list)
    for i in range(len(tokens)-n+1):
        key  = tuple(tokens[i:i+n-1]) if n>1 else ()
        model[key].append(tokens[i+n-1])
    return model

def gen_ngram(model, n, start, length=20, seed=0):
    random.seed(seed)
    if n == 1:
        result = [random.choice(tokens)]
        for _ in range(length-1):
            nxt = model.get((), [])
            if not nxt: break
            result.append(random.choice(nxt))
        return " ".join(result)
    result = list(start[:n-1])
    for _ in range(length-len(result)):
        key  = tuple(result[-(n-1):])
        nxt  = model.get(key, [])
        if not nxt: break
        result.append(random.choice(nxt))
    return " ".join(result)

for n, start in [(1,()), (2,("the",)), (3,("the","quick"))]:
    m = build_ngram(tokens, n)
    g = gen_ngram(m, n, start)
    print(f"  n={n}: {g}")

# Ex 2
print("\n=== Ex 2: Temperature Sampling ===")
bigram_model = defaultdict(list)
for i in range(len(tokens)-1):
    bigram_model[tokens[i]].append(tokens[i+1])

def gen_temp(start, n_words=15, temp=1.0, seed=0):
    np.random.seed(seed)
    word, result = start, [start]
    for _ in range(n_words-1):
        nexts = bigram_model.get(word, [])
        if not nexts: break
        counts = np.array([nexts.count(w) for w in set(nexts)], dtype=float)
        words  = list(set(nexts))
        log_p  = np.log(counts+1e-10) / temp
        log_p -= log_p.max()
        probs  = np.exp(log_p); probs /= probs.sum()
        word   = np.random.choice(words, p=probs)
        result.append(word)
    return " ".join(result)

for temp in [0.1, 1.0, 2.0]:
    print(f"  temp={temp}: {gen_temp('the', temp=temp)}")

# Ex 3
print("\n=== Ex 3: Sentence Score Visualization ===")
TEXT = """
Scientists developed a new battery technology at MIT.
The lithium-sulfur battery achieves three times higher energy density.
Sulfur is cheap and abundant compared to lithium-ion materials.
A conductive polymer coating prevents degradation of sulfur cathodes.
Tests showed 80 percent capacity retention after 500 charge cycles.
Commercial applications include electric trucks and aviation.
The team expects pilot manufacturing within two years.
Analysts estimate costs will fall by 60 percent.
"""
sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", TEXT.strip()) if s.strip()]
vec   = TfidfVectorizer(stop_words="english").fit_transform(sents)
scores= vec.sum(axis=1).A1

fig, ax = plt.subplots(figsize=(8,4))
ax.barh(range(len(sents)), scores, color="steelblue", alpha=0.8)
ax.set_yticks(range(len(sents)))
ax.set_yticklabels([s[:40] for s in sents], fontsize=7)
ax.set_xlabel("TF-IDF Score"); ax.set_title("Sentence Importance Scores")
plt.tight_layout(); plt.savefig("sentence_scores.png",dpi=80); plt.close(); print("  Saved sentence_scores.png")
top3 = [sents[i] for i in np.argsort(scores)[-3:][::-1]]
print(f"  Top 3: {top3}")

# Ex 4
print("\n=== Ex 4: Centroid-Based Summarization ===")
def centroid_summarize(text, n=3):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    vec   = TfidfVectorizer(stop_words="english").fit_transform(sents)
    centroid = vec.mean(axis=0)
    sims = cosine_similarity(vec, centroid).flatten()
    top  = sorted(np.argsort(sims)[-n:])
    return " ".join(sents[i] for i in top)
print("  "+centroid_summarize(TEXT))

# Ex 5
print("\n=== Ex 5: Mad-Lib Generator ===")
VOCAB = {
    "ADJ":["fluffy","ancient","enormous","suspicious","transparent"],
    "NOUN":["robot","penguin","umbrella","professor","spaceship"],
    "VERB_PAST":["exploded","danced","snored","teleported","melted"],
    "PLACE":["Paris","the basement","a volcano","the moon","Nebraska"],
}
TEMPLATES_ML = [
    "The {ADJ} {NOUN} {VERB_PAST} through {PLACE} and scared everyone.",
    "Yesterday, a {NOUN} {VERB_PAST} on a {ADJ} mountain in {PLACE}.",
]
random.seed(42)
for _ in range(5):
    t = random.choice(TEMPLATES_ML)
    for slot, choices in VOCAB.items():
        t = t.replace("{"+slot+"}", random.choice(choices))
    print(f"  {t}")
