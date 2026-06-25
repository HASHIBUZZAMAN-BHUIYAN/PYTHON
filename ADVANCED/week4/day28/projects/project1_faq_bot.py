"""
Project: FAQ Chatbot with Topic Routing
Teaches: retrieval chatbot with topic detection, multi-KB routing,
         confidence display and unknown rejection.
~30 MB RAM, ~1s on CPU
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Topic-specific knowledge bases ──────────────────────────────────────────
KB_TECH = [
    ("What is Python?",           "Python is a versatile, high-level language popular for data science and web dev."),
    ("What is machine learning?", "ML systems learn patterns from data without explicit programming."),
    ("What is a GPU?",            "GPUs are parallel processors originally for graphics, now essential for deep learning."),
    ("What is an API?",           "An API (Application Programming Interface) lets software programs communicate."),
    ("What is cloud computing?",  "Cloud computing delivers computing resources (servers, storage) over the internet."),
    ("What is open source?",      "Open source software has publicly available code that anyone can use or modify."),
]
KB_SCIENCE = [
    ("What is photosynthesis?",   "Photosynthesis converts light, water and CO2 into glucose and oxygen in plants."),
    ("What is DNA?",              "DNA is the molecule carrying genetic instructions for all living organisms."),
    ("What is gravity?",          "Gravity is the force attracting objects with mass toward each other."),
    ("What is a black hole?",     "A black hole is a region in space where gravity is so strong not even light escapes."),
    ("What is evolution?",        "Evolution is the change in species' heritable traits across generations by natural selection."),
]
KB_GENERAL = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("How many planets in solar system?", "There are 8 planets in our solar system."),
    ("What is the speed of light?",   "The speed of light is approximately 299,792,458 meters per second."),
    ("Who wrote Hamlet?",             "Hamlet was written by William Shakespeare around 1600."),
    ("What is the boiling point of water?", "Water boils at 100°C (212°F) at standard atmospheric pressure."),
]

ALL_KBS  = {"Tech": KB_TECH, "Science": KB_SCIENCE, "General": KB_GENERAL}

# ─── Build TF-IDF index per topic ─────────────────────────────────────────────
def preprocess(t):
    return re.sub(r"[^\w\s]", "", t.lower()).strip()

class FAQBot:
    def __init__(self, kbs):
        self.kbs = kbs
        self.indexes = {}
        for topic, pairs in kbs.items():
            qs = [preprocess(q) for q, _ in pairs]
            ans= [a for _, a in pairs]
            v  = TfidfVectorizer(ngram_range=(1,2), max_features=300)
            X  = v.fit_transform(qs)
            self.indexes[topic] = {"vec": v, "X": X, "answers": ans, "questions": qs}

    def ask(self, query, threshold=0.08):
        q_pre = preprocess(query)
        best_topic, best_score, best_ans = None, -1, None
        for topic, idx in self.indexes.items():
            q_vec = idx["vec"].transform([q_pre])
            sims  = cosine_similarity(q_vec, idx["X"]).flatten()
            if sims.max() > best_score:
                best_score = float(sims.max())
                best_topic = topic
                best_ans   = idx["answers"][int(np.argmax(sims))]
        if best_score < threshold:
            return "I don't have information about that topic.", "Unknown", 0.0
        return best_ans, best_topic, best_score

bot = FAQBot(ALL_KBS)

QUERIES = [
    "How does Python programming work?",
    "Tell me about black holes in space.",
    "What is the capital city of France?",
    "How do neural networks learn?",
    "What is photosynthesis?",
    "Can you explain cloud computing?",
    "Who is the best football player?",  # OOD
    "What is DNA made of?",
]

print("=== FAQ Chatbot with Topic Routing ===\n")
print(f"{'Query':<48}  {'Topic':>10}  {'Conf':>6}")
print("─"*70)
for q in QUERIES:
    answer, topic, conf = bot.ask(q)
    print(f"  {q:<46}  {topic:>10}  {conf:>5.3f}")
    print(f"     → {answer}\n")
