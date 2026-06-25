# AGENTIC_AI Reference — Agent Memory Template
# Short-term (conversation buffer), long-term (JSON file),
# and vector memory (TF-IDF similarity search).
# ~15 MB RAM, <1s on CPU

import json
import os
import numpy as np
from collections import deque

# ─── 1. CONVERSATION BUFFER (short-term) ─────────────────────────────────────
class ConversationBuffer:
    """
    Rolling window of the last N conversation turns.
    Each turn is a dict: {"role": "user"/"assistant", "content": str}
    """
    def __init__(self, max_turns: int = 5):
        self.buffer   = deque(maxlen=max_turns * 2)  # 2 messages per turn
        self.max_turns= max_turns

    def add(self, role: str, content: str):
        self.buffer.append({"role": role, "content": content})

    def get_context(self) -> list:
        return list(self.buffer)

    def format_context(self) -> str:
        lines = []
        for msg in self.buffer:
            prefix = "User" if msg["role"]=="user" else "Assistant"
            lines.append(f"{prefix}: {msg['content']}")
        return "\n".join(lines)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

# ─── 2. JSON FACT MEMORY (long-term) ──────────────────────────────────────────
class FactMemory:
    """
    Persistent key-value fact store backed by a JSON file.
    Keys and values are strings.
    """
    def __init__(self, filepath: str = "fact_memory.json"):
        self.filepath = filepath
        self._data: dict = {}
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath) as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def _save(self):
        with open(self.filepath, "w") as f:
            json.dump(self._data, f, indent=2)

    def store(self, key: str, value: str):
        self._data[key.lower().strip()] = value
        self._save()

    def recall(self, key: str) -> str | None:
        return self._data.get(key.lower().strip())

    def recall_fuzzy(self, query: str, top_k=3) -> list:
        """Return top-k facts whose key overlaps with query words."""
        q_words = set(query.lower().split())
        scores  = [(k, len(set(k.split()) & q_words)) for k in self._data]
        scores  = [(k,s) for k,s in scores if s>0]
        scores.sort(key=lambda x:-x[1])
        return [(k, self._data[k]) for k,_ in scores[:top_k]]

    def list_all(self) -> dict:
        return dict(self._data)

    def delete(self, key: str) -> bool:
        key = key.lower().strip()
        if key in self._data:
            del self._data[key]; self._save(); return True
        return False

    def clear(self):
        self._data = {}; self._save()

# ─── 3. TFIDF VECTOR MEMORY (semantic retrieval) ──────────────────────────────
class VectorMemory:
    """
    Store (question, answer) pairs and retrieve by TF-IDF cosine similarity.
    Useful for FAQ-style memory or experience replay.
    """
    def __init__(self, max_entries: int = 200):
        self.max_entries = max_entries
        self.memories: list = []    # list of {"q": str, "a": str}
        self._vectorizer = None
        self._matrix     = None
        self._dirty      = True

    def add(self, question: str, answer: str):
        self.memories.append({"q": question, "a": answer})
        if len(self.memories) > self.max_entries:
            self.memories.pop(0)
        self._dirty = True

    def _build_index(self):
        if not self._dirty: return
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [m["q"] for m in self.memories]
        if not texts: return
        self._vectorizer = TfidfVectorizer(max_features=1000)
        self._matrix     = self._vectorizer.fit_transform(texts).toarray()
        self._dirty      = False

    def search(self, query: str, k: int = 3) -> list:
        """Returns list of (question, answer, score) sorted by relevance."""
        self._build_index()
        if self._vectorizer is None: return []
        q_vec = self._vectorizer.transform([query]).toarray()[0]
        norms = np.linalg.norm(self._matrix, axis=1) * np.linalg.norm(q_vec) + 1e-9
        sims  = self._matrix.dot(q_vec) / norms
        top   = np.argsort(sims)[::-1][:k]
        return [(self.memories[i]["q"], self.memories[i]["a"], float(sims[i]))
                for i in top if sims[i] > 0]

    def __len__(self): return len(self.memories)

# ─── 4. UNIFIED MEMORY MANAGER ───────────────────────────────────────────────
class AgentMemory:
    """
    Combines short-term buffer + long-term facts + vector retrieval.
    """
    def __init__(self, buffer_turns=5, fact_file="agent_facts.json"):
        self.buffer = ConversationBuffer(buffer_turns)
        self.facts  = FactMemory(fact_file)
        self.vector = VectorMemory()

    def remember_fact(self, key: str, value: str):
        self.facts.store(key, value)
        self.vector.add(key, value)

    def recall_fact(self, query: str, top_k=3):
        exact = self.facts.recall(query)
        if exact: return [(query, exact, 1.0)]
        return self.vector.search(query, k=top_k)

    def add_turn(self, user_msg: str, assistant_msg: str):
        self.buffer.add("user", user_msg)
        self.buffer.add("assistant", assistant_msg)

    def get_context_string(self) -> str:
        return self.buffer.format_context()

# ─── DEMO ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== ConversationBuffer ===")
    buf = ConversationBuffer(max_turns=3)
    for i in range(5):
        buf.add("user",      f"User message {i+1}")
        buf.add("assistant", f"Assistant reply {i+1}")
    print(f"  Buffer size (max 6 msgs): {len(buf)}")
    print(f"  Context:\n{buf.format_context()}")

    print("\n=== FactMemory ===")
    fm = FactMemory("demo_facts.json")
    fm.clear()
    fm.store("user name",    "Alex")
    fm.store("user language","Python")
    fm.store("project name", "CV pipeline")
    print(f"  Recalled 'user name': {fm.recall('user name')}")
    fuzzy = fm.recall_fuzzy("what is the project?")
    print(f"  Fuzzy 'project': {fuzzy}")
    os.remove("demo_facts.json")

    print("\n=== VectorMemory ===")
    vm = VectorMemory()
    qa_pairs = [
        ("What is Python?", "Python is a high-level programming language."),
        ("What is machine learning?", "ML enables systems to learn from data."),
        ("What is deep learning?", "Deep learning uses neural networks with many layers."),
        ("What is PyTorch?", "PyTorch is a CPU/GPU deep learning framework."),
    ]
    for q, a in qa_pairs: vm.add(q, a)
    results = vm.search("neural network framework", k=2)
    for q, a, s in results:
        print(f"  score={s:.3f}  Q: {q}  A: {a[:40]}")
