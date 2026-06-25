# ~120 MB RAM, ~2s on CPU
"""
Day 36 — Agent Memory: Solutions
==================================
Full working solutions for all 5 exercises.
"""

import json
import os
import time
from collections import deque

# ──────────────────────────────────────────────
# Exercise 1 — Conversation Buffer with Overflow
# ──────────────────────────────────────────────
print("=" * 60)
print("EXERCISE 1 — Conversation Buffer (OpenAI message format)")
print("=" * 60)


class ConversationBuffer:
    """Stores messages as {'role': ..., 'content': ...} dicts.
    Keeps at most max_turns user+assistant pairs (2*max_turns messages)."""

    def __init__(self, max_turns: int = 4):
        self.max_turns = max_turns
        # Each slot = (user_msg, assistant_msg)
        self._pairs: deque = deque(maxlen=max_turns)

    def add(self, user_content: str, assistant_content: str) -> None:
        self._pairs.append((user_content, assistant_content))

    def messages(self) -> list:
        """Return flat OpenAI-style message list."""
        out = []
        for u, a in self._pairs:
            out.append({"role": "user", "content": u})
            out.append({"role": "assistant", "content": a})
        return out

    def __len__(self):
        # Number of messages (pairs * 2)
        return len(self._pairs) * 2


buf = ConversationBuffer(max_turns=4)
turns_data = [
    ("What is 1+1?", "2."),
    ("Name a planet.", "Mars."),
    ("Capital of Japan?", "Tokyo."),
    ("What colour is grass?", "Green."),
    ("How many legs does a cat have?", "Four."),
    ("What is Python?", "Python is a programming language."),
]
for u, a in turns_data:
    buf.add(u, a)

msgs = buf.messages()
assert len(msgs) == 8, f"Expected 8 messages, got {len(msgs)}"
print(f"Buffer holds {len(buf)} messages (4 pairs, oldest 2 turns evicted):")
for m in msgs:
    print(f"  {m['role']:>9}: {m['content']}")


# ──────────────────────────────────────────────
# Exercise 2 — Key-Value Store with Expiry
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXERCISE 2 — Key-Value Store with TTL Expiry")
print("=" * 60)


class KeyValueMemory:
    """In-memory key-value store with per-entry TTL."""

    def __init__(self, default_ttl: float = 60.0):
        self.default_ttl = default_ttl
        self._store: dict = {}  # key → (value, expires_at)

    def store(self, key: str, value: str, ttl: float = None) -> None:
        ttl = ttl if ttl is not None else self.default_ttl
        expires_at = time.time() + ttl
        self._store[key] = (value, expires_at)
        print(f"  STORE  {key!r} = {value!r}  (expires in {ttl}s)")

    def recall(self, key: str):
        if key not in self._store:
            return None
        value, expires_at = self._store[key]
        if time.time() > expires_at:
            del self._store[key]
            print(f"  RECALL {key!r} → EXPIRED (removed)")
            return None
        print(f"  RECALL {key!r} → {value!r}")
        return value

    def list_all(self) -> list:
        now = time.time()
        return [(k, v) for k, (v, exp) in self._store.items() if exp > now]


kv = KeyValueMemory(default_ttl=60.0)
kv.store("long_lived", "I will survive", ttl=60)
kv.store("short_lived", "I expire fast", ttl=1)

print("  Recalling before expiry:")
kv.recall("long_lived")
kv.recall("short_lived")

print("  Sleeping 2 seconds...")
time.sleep(2)

print("  Recalling after expiry:")
kv.recall("long_lived")   # still valid
kv.recall("short_lived")  # expired


# ──────────────────────────────────────────────
# Exercise 3 — TF-IDF Memory Search
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXERCISE 3 — TF-IDF Memory Search")
print("=" * 60)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN = True
except ImportError:
    SKLEARN = False


class TFIDFMemory:
    def __init__(self):
        self._docs: list = []
        self._vectorizer = None
        self._matrix = None
        self._dirty = True

    def add(self, text: str) -> None:
        self._docs.append(text)
        self._dirty = True

    def _fit(self):
        if SKLEARN:
            self._vectorizer = TfidfVectorizer(stop_words="english")
            self._matrix = self._vectorizer.fit_transform(self._docs)
        self._dirty = False

    def search(self, query: str, top_k: int = 2):
        if self._dirty:
            self._fit()
        if not self._docs:
            return []
        if SKLEARN:
            q_vec = self._vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self._matrix)[0]
            ranked = sorted(enumerate(sims), key=lambda x: -x[1])
            return [(self._docs[i], float(s)) for i, s in ranked[:top_k]]
        else:
            q_words = set(query.lower().split())
            results = []
            for doc in self._docs:
                d_words = set(doc.lower().split())
                union = q_words | d_words
                score = len(q_words & d_words) / len(union) if union else 0.0
                results.append((doc, score))
            results.sort(key=lambda x: -x[1])
            return results[:top_k]


knowledge = [
    "The sun is a star at the center of our solar system.",
    "Water freezes at 0 degrees Celsius.",
    "Python is an interpreted high-level programming language.",
    "Machine learning algorithms learn patterns from data.",
    "DNA carries the genetic information of living organisms.",
    "The speed of light is approximately 299,792,458 meters per second.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "Neural networks are composed of layers of artificial neurons.",
    "Gravity is the force that attracts two bodies with mass.",
    "The Eiffel Tower is located in Paris, France.",
]

mem = TFIDFMemory()
for s in knowledge:
    mem.add(s)

test_queries = [
    "How does machine learning work?",
    "What is the temperature at which water becomes ice?",
    "Tell me about programming languages.",
]

for q in test_queries:
    print(f"\n  Query: {q!r}")
    for sentence, score in mem.search(q, top_k=2):
        print(f"    score={score:.3f}  {sentence[:70]}")


# ──────────────────────────────────────────────
# Exercise 4 — Memory Consolidation Agent
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXERCISE 4 — Memory Consolidation Agent")
print("=" * 60)


class ConsolidatingAgent:
    def __init__(self, max_turns: int = 3):
        self.max_turns = max_turns
        self.buffer: list = []
        self.long_term: dict = {}

    def _make_key(self, text: str) -> str:
        words = text.lower().split()
        return "_".join(words[:3])

    def add(self, user: str, assistant: str) -> None:
        if len(self.buffer) >= self.max_turns:
            evicted = self.buffer.pop(0)
            key = self._make_key(evicted["user"])
            value = evicted["assistant"][:50]
            self.long_term[key] = value
            print(f"  [CONSOLIDATE] '{key}' ← '{value}'")
        self.buffer.append({"user": user, "assistant": assistant})

    def status(self):
        print(f"  Buffer: {[t['user'][:25] for t in self.buffer]}")
        print(f"  LT Store: {self.long_term}")


agent_turns = [
    ("What is Python?", "Python is an interpreted programming language."),
    ("Who created Python?", "Guido van Rossum created Python in 1991."),
    ("Is Python fast?", "Python is slower than compiled languages."),
    ("What are Python lists?", "Lists are ordered mutable sequences."),
    ("What are Python dicts?", "Dicts are key-value mappings."),
    ("What is a function?", "A function is a reusable block of code."),
    ("What is a class?", "A class is a blueprint for objects."),
    ("What is inheritance?", "Inheritance lets a class extend another."),
]

ca = ConsolidatingAgent(max_turns=3)
for u, a in agent_turns:
    print(f"\n  Adding: U={u!r}")
    ca.add(u, a)
    ca.status()

print(f"\n  Final long-term store ({len(ca.long_term)} entries):")
for k, v in ca.long_term.items():
    print(f"    {k!r}: {v!r}")


# ──────────────────────────────────────────────
# Exercise 5 — Multi-Type Memory Manager
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXERCISE 5 — Multi-Type Memory Manager")
print("=" * 60)


class MemoryManager:
    def __init__(self, max_turns: int = 5):
        self._buffer: deque = deque(maxlen=max_turns)
        self._kv: dict = {}
        self._tfidf = TFIDFMemory()

    def remember(self, user: str, assistant: str) -> None:
        self._buffer.append({"user": user, "assistant": assistant})
        for sentence in assistant.replace("!", ".").replace("?", ".").split("."):
            sentence = sentence.strip()
            if len(sentence) > 10:
                self._tfidf.add(sentence)

    def kv_store(self, key: str, value: str) -> None:
        self._kv[key] = value

    def retrieve(self, query: str) -> tuple:
        results = self._tfidf.search(query, top_k=1)
        if results:
            text, score = results[0]
            if score >= 0.1:
                return ("tfidf", text, score)
        # Fallback: keyword match in KV
        q_lower = query.lower()
        for k, v in self._kv.items():
            if k.lower() in q_lower:
                return ("kv", f"{k}: {v}", 1.0)
        return ("none", None, 0.0)


mm = MemoryManager(max_turns=5)
mm.kv_store("python_version", "3.12 is latest stable")

conversation = [
    ("What is Python?", "Python is a high-level programming language. It is easy to learn."),
    ("Who made it?", "Guido van Rossum created Python. He started it in 1989."),
    ("Why use Python?", "Python is readable and has many libraries. It is great for data science."),
    ("What are lists?", "Lists are ordered mutable sequences. You can append and remove items."),
    ("What are dicts?", "Dicts map keys to values. They are unordered in Python 2 but ordered in Python 3."),
]

for u, a in conversation:
    mm.remember(u, a)

for query in ["machine learning Python", "who invented Python", "python_version latest"]:
    source, text, score = mm.retrieve(query)
    print(f"\n  Query: {query!r}")
    print(f"    Source: {source}, Score: {score:.3f}")
    print(f"    Result: {str(text)[:80]}")

print("\n--- Solutions complete ---")
