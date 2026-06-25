# ~120 MB RAM, ~2s on CPU
"""
Day 36 — Agent Memory
=====================
Four memory mechanisms used in agentic AI systems:
  1. Short-term / Conversation Buffer
  2. Long-term / Key-Value JSON Store
  3. Vector Memory via TF-IDF cosine similarity
  4. Memory Consolidation (buffer → long-term)
"""

import json
import math
import os
import time
from collections import deque

# ─────────────────────────────────────────────────────────────
# SECTION 1 — SHORT-TERM MEMORY: CONVERSATION BUFFER
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 1 — CONVERSATION BUFFER (SHORT-TERM MEMORY)")
print("=" * 60)

class ConversationBuffer:
    """
    Stores the last `max_turns` (user, assistant) pairs.
    Once full, the oldest turn is discarded (FIFO).
    Agents use this to maintain context without growing forever.
    """

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.buffer: deque = deque(maxlen=max_turns)

    def add(self, user_msg: str, assistant_msg: str) -> None:
        self.buffer.append({"user": user_msg, "assistant": assistant_msg})

    def get_context(self) -> list:
        """Return buffer as a plain list (newest last)."""
        return list(self.buffer)

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        lines = [f"ConversationBuffer(max={self.max_turns}, used={len(self)})"]
        for i, turn in enumerate(self.buffer, 1):
            lines.append(f"  [{i}] U: {turn['user']!r}  A: {turn['assistant']!r}")
        return "\n".join(lines)


buf = ConversationBuffer(max_turns=3)
buf.add("Hi!", "Hello, how can I help?")
buf.add("What is Python?", "Python is a high-level programming language.")
buf.add("Who created it?", "Guido van Rossum created Python in 1991.")
buf.add("Is it fast?", "It's slower than C but fast to write and read.")

print(buf)
print(f"\nBuffer length (should be 3, oldest dropped): {len(buf)}")
print(f"Context: {buf.get_context()}")


# ─────────────────────────────────────────────────────────────
# SECTION 2 — LONG-TERM MEMORY: KEY-VALUE JSON STORE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2 — KEY-VALUE JSON STORE (LONG-TERM MEMORY)")
print("=" * 60)

class KeyValueMemory:
    """
    Persistent facts stored in a JSON file.
    Supports STORE, RECALL, LIST, DELETE operations.
    """

    def __init__(self, filepath: str = "memory_store.json"):
        self.filepath = filepath
        self._store: dict = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                self._store = json.load(f)

    def _save(self) -> None:
        with open(self.filepath, "w") as f:
            json.dump(self._store, f, indent=2)

    def store(self, key: str, value: str) -> str:
        self._store[key] = value
        self._save()
        return f"Stored: {key!r} = {value!r}"

    def recall(self, key: str) -> str:
        if key in self._store:
            return f"Recalled: {key!r} = {self._store[key]!r}"
        return f"Not found: {key!r}"

    def list_all(self) -> list:
        return list(self._store.items())

    def delete(self, key: str) -> str:
        if key in self._store:
            del self._store[key]
            self._save()
            return f"Deleted: {key!r}"
        return f"Key not found: {key!r}"

    def __len__(self):
        return len(self._store)


# Demo — use a temp file in the current directory
kv = KeyValueMemory("lesson_demo_memory.json")
print(kv.store("capital_of_france", "Paris"))
print(kv.store("pi", "3.14159"))
print(kv.store("python_creator", "Guido van Rossum"))
print(kv.recall("capital_of_france"))
print(kv.recall("unknown_key"))
print(f"All facts ({len(kv)}):", kv.list_all())
print(kv.delete("pi"))
# Clean up demo file
try:
    os.remove("lesson_demo_memory.json")
except FileNotFoundError:
    pass


# ─────────────────────────────────────────────────────────────
# SECTION 3 — VECTOR MEMORY: TF-IDF COSINE SIMILARITY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3 — VECTOR MEMORY (TF-IDF COSINE SIMILARITY)")
print("=" * 60)

"""
Why TF-IDF instead of dense embeddings?
  - No GPU required, runs on any machine
  - scikit-learn TfidfVectorizer fits in <50 MB RAM for thousands of docs
  - Cosine similarity is fast (sparse dot product)
  - Good enough for keyword-heavy FAQ / fact retrieval
"""

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("  [NOTE] scikit-learn not installed — showing manual TF-IDF fallback.\n")


class TFIDFMemory:
    """
    Stores (text, metadata) pairs.
    On query, returns the top-k most similar entries.
    """

    def __init__(self):
        self._docs: list = []          # raw text
        self._meta: list = []          # paired metadata (any object)
        self._vectorizer = None
        self._matrix = None
        self._dirty = True             # needs re-fit when docs change

    def add(self, text: str, meta=None) -> None:
        self._docs.append(text)
        self._meta.append(meta)
        self._dirty = True

    def _fit(self) -> None:
        if not self._docs:
            return
        if SKLEARN_AVAILABLE:
            self._vectorizer = TfidfVectorizer(stop_words="english")
            self._matrix = self._vectorizer.fit_transform(self._docs)
        self._dirty = False

    def query(self, text: str, top_k: int = 3):
        if self._dirty:
            self._fit()
        if not self._docs:
            return []

        if SKLEARN_AVAILABLE:
            q_vec = self._vectorizer.transform([text])
            sims = cosine_similarity(q_vec, self._matrix)[0]
            ranked = sorted(enumerate(sims), key=lambda x: -x[1])
            results = []
            for idx, score in ranked[:top_k]:
                results.append((self._docs[idx], self._meta[idx], float(score)))
            return results
        else:
            # Simple fallback: word overlap ratio
            q_words = set(text.lower().split())
            results = []
            for i, doc in enumerate(self._docs):
                d_words = set(doc.lower().split())
                if q_words | d_words:
                    score = len(q_words & d_words) / len(q_words | d_words)
                else:
                    score = 0.0
                results.append((doc, self._meta[i], score))
            results.sort(key=lambda x: -x[2])
            return results[:top_k]


mem = TFIDFMemory()
facts = [
    ("Python is a high-level interpreted programming language.", "python_basics"),
    ("Machine learning uses algorithms to learn from data.", "ml_basics"),
    ("Neural networks are inspired by the human brain.", "neural_nets"),
    ("TF-IDF stands for Term Frequency-Inverse Document Frequency.", "tfidf"),
    ("Cosine similarity measures the angle between two vectors.", "cosine_sim"),
    ("A* is a graph search algorithm that uses heuristics.", "a_star"),
    ("PID controllers use proportional integral derivative terms.", "pid"),
]
for text, tag in facts:
    mem.add(text, tag)

queries = [
    "How does machine learning work?",
    "What is cosine similarity used for?",
    "Tell me about neural networks",
]

for q in queries:
    print(f"\nQuery: {q!r}")
    for doc, tag, score in mem.query(q, top_k=2):
        print(f"  [{tag}] score={score:.3f}  {doc[:60]}")


# ─────────────────────────────────────────────────────────────
# SECTION 4 — MEMORY CONSOLIDATION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 4 — MEMORY CONSOLIDATION")
print("=" * 60)

"""
Consolidation strategy:
  When the conversation buffer fills, "summarise" the oldest turn
  into the long-term key-value store, then evict it from the buffer.

  In production: a real LLM does the summarisation.
  Here: mock summariser extracts the user question as the key
        and the first sentence of the assistant answer as the value.
"""


def mock_summarise(turn: dict) -> tuple:
    """Return (key, summary) from a single conversation turn."""
    key = turn["user"].strip("?!. ").lower().replace(" ", "_")[:40]
    answer = turn["assistant"].split(".")[0].strip()
    return key, answer


class ConsolidatingMemory:
    """
    Short-term buffer + long-term KV store.
    When buffer hits capacity, oldest turn is consolidated into LT.
    """

    def __init__(self, max_turns: int = 3, kv_path: str = None):
        self.buffer = []               # list of dicts
        self.max_turns = max_turns
        self.lt_store: dict = {}       # in-memory long-term store
        self.kv_path = kv_path

    def add(self, user_msg: str, assistant_msg: str) -> None:
        if len(self.buffer) >= self.max_turns:
            oldest = self.buffer.pop(0)
            key, summary = mock_summarise(oldest)
            self.lt_store[key] = summary
            print(f"  [CONSOLIDATE] '{key}' → '{summary}'")
        self.buffer.append({"user": user_msg, "assistant": assistant_msg})

    def get_context(self) -> list:
        return list(self.buffer)

    def recall_lt(self, key: str) -> str:
        return self.lt_store.get(key, "(not found)")

    def status(self):
        print(f"  Buffer ({len(self.buffer)}/{self.max_turns}): "
              f"{[t['user'][:20] for t in self.buffer]}")
        print(f"  Long-term store ({len(self.lt_store)} entries): "
              f"{list(self.lt_store.keys())}")


cMemory = ConsolidatingMemory(max_turns=3)
turns = [
    ("What is Python?", "Python is a high-level programming language."),
    ("Who made Python?", "Guido van Rossum created Python in 1991."),
    ("Is Python fast?", "Python is slower than C but very readable."),
    ("What is a list?", "A list is an ordered mutable collection in Python."),
    ("What is a dict?", "A dict is a key-value mapping in Python."),
]

for user, asst in turns:
    print(f"\nAdding turn: U={user!r}")
    cMemory.add(user, asst)
    cMemory.status()

print("\nLong-term store:", cMemory.lt_store)
print("\n--- Lesson complete ---")
