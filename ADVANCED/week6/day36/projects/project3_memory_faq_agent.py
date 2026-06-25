# ~150 MB RAM, ~3s on CPU
"""
Project 3 — Memory FAQ Agent (TF-IDF Search)
=============================================
An FAQ agent that searches a knowledge base using TF-IDF cosine similarity.
- 15 hardcoded Q&A pairs loaded as the knowledge base.
- Queries matched by cosine similarity (sklearn) or word-overlap fallback.
- Demonstrates: search for 5 test questions + add a new fact then query it.
- No API key required.
"""

# ──────────────────────────────────────────────
# TF-IDF IMPORTS (with offline fallback)
# ──────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ──────────────────────────────────────────────
# KNOWLEDGE BASE — 15 Q&A pairs
# ──────────────────────────────────────────────

FAQ_KB: list[dict] = [
    {"q": "What is Python?",
     "a": "Python is a high-level, interpreted, general-purpose programming language created by Guido van Rossum in 1991."},
    {"q": "What is machine learning?",
     "a": "Machine learning is a branch of AI where algorithms learn patterns from data without being explicitly programmed."},
    {"q": "What is a neural network?",
     "a": "A neural network is a machine learning model inspired by the human brain, composed of layers of interconnected nodes."},
    {"q": "What is TF-IDF?",
     "a": "TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that reflects the importance of a word in a document relative to a corpus."},
    {"q": "What is cosine similarity?",
     "a": "Cosine similarity measures the cosine of the angle between two vectors; a value of 1 means identical direction, 0 means orthogonal."},
    {"q": "What is an AI agent?",
     "a": "An AI agent is a system that perceives its environment, reasons about goals, and takes actions to achieve them."},
    {"q": "What is the A* algorithm?",
     "a": "A* is a best-first graph search algorithm that uses a heuristic to efficiently find the shortest path between two nodes."},
    {"q": "What is a PID controller?",
     "a": "A PID controller uses proportional, integral, and derivative terms to regulate a system toward a desired setpoint."},
    {"q": "What is memory in AI?",
     "a": "AI memory stores information from past interactions; short-term memory holds recent context, long-term memory persists across sessions."},
    {"q": "What is RAG?",
     "a": "Retrieval-Augmented Generation (RAG) combines a retrieval system with a generative model to ground responses in external documents."},
    {"q": "What is a large language model?",
     "a": "A large language model (LLM) is a neural network trained on vast amounts of text to understand and generate human language."},
    {"q": "What is reinforcement learning?",
     "a": "Reinforcement learning trains an agent by rewarding desired behaviours and penalising undesired ones through environment interaction."},
    {"q": "What is overfitting?",
     "a": "Overfitting occurs when a model learns the training data too closely and performs poorly on unseen data."},
    {"q": "What is gradient descent?",
     "a": "Gradient descent is an optimisation algorithm that iteratively adjusts model parameters to minimise a loss function."},
    {"q": "What is tokenisation?",
     "a": "Tokenisation splits text into smaller units (tokens) such as words or subwords for processing by NLP models."},
]

# ──────────────────────────────────────────────
# FAQ AGENT
# ──────────────────────────────────────────────

class FAQAgent:
    """
    Searches a Q&A knowledge base using TF-IDF cosine similarity.
    Falls back to word-overlap Jaccard score if sklearn is unavailable.
    Supports dynamically adding new facts at runtime.
    """

    def __init__(self, kb: list[dict]):
        # Store everything as the question text; answer linked by index
        self._questions: list[str] = [item["q"] for item in kb]
        self._answers:   list[str] = [item["a"] for item in kb]
        self._vectorizer = None
        self._matrix = None
        self._dirty = True  # needs re-fit when KB changes

    def _fit(self) -> None:
        if not self._questions:
            return
        if SKLEARN_AVAILABLE:
            self._vectorizer = TfidfVectorizer(stop_words="english")
            self._matrix = self._vectorizer.fit_transform(self._questions)
        self._dirty = False

    def add_fact(self, question: str, answer: str) -> None:
        self._questions.append(question)
        self._answers.append(answer)
        self._dirty = True
        print(f"  [ADD] New fact stored: {question!r}")

    def search(self, query: str, top_k: int = 1) -> list[dict]:
        """Return list of dicts with keys: question, answer, score."""
        if self._dirty:
            self._fit()
        if not self._questions:
            return []

        if SKLEARN_AVAILABLE:
            q_vec = self._vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self._matrix)[0]
            ranked = sorted(enumerate(sims), key=lambda x: -x[1])
            results = []
            for idx, score in ranked[:top_k]:
                results.append({
                    "question": self._questions[idx],
                    "answer":   self._answers[idx],
                    "score":    float(score),
                })
            return results
        else:
            # Word-overlap fallback
            q_words = set(query.lower().split())
            results = []
            for i, qtext in enumerate(self._questions):
                d_words = set(qtext.lower().split())
                union = q_words | d_words
                score = len(q_words & d_words) / len(union) if union else 0.0
                results.append({
                    "question": qtext,
                    "answer":   self._answers[i],
                    "score":    score,
                })
            results.sort(key=lambda x: -x["score"])
            return results[:top_k]

    def answer(self, query: str) -> None:
        """Pretty-print the best answer for a query."""
        results = self.search(query, top_k=1)
        if not results or results[0]["score"] < 0.01:
            print(f"  Answer : (no relevant answer found)")
            return
        r = results[0]
        print(f"  Match  : {r['question']!r}")
        print(f"  Score  : {r['score']:.4f}")
        print(f"  Answer : {r['answer']}")


# ──────────────────────────────────────────────
# MAIN DEMO
# ──────────────────────────────────────────────

def run_demo():
    engine_label = "sklearn TF-IDF" if SKLEARN_AVAILABLE else "word-overlap fallback"
    print("=" * 70)
    print(f"  FAQ AGENT — TF-IDF Memory Search  (engine: {engine_label})")
    print(f"  KB size: {len(FAQ_KB)} Q&A pairs")
    print("=" * 70)

    agent = FAQAgent(FAQ_KB)

    # ── 5 test questions ─────────────────────────────────────────────────

    test_queries = [
        "How does machine learning work?",
        "What is cosine similarity used for in text?",
        "Tell me about neural networks and the brain.",
        "How does an AI agent perceive and act?",
        "What does TF-IDF measure?",
    ]

    print("\n  ── 5 Test Queries ──────────────────────────────────────────────\n")
    for i, q in enumerate(test_queries, 1):
        print(f"  [{i}] Query: {q!r}")
        agent.answer(q)
        print()

    # ── Add a new fact and immediately query it ──────────────────────────

    print("  ── Adding a new fact at runtime ────────────────────────────────\n")
    agent.add_fact(
        question="What is quantisation in machine learning?",
        answer="Quantisation reduces the precision of model weights (e.g., float32 → int8) to decrease memory usage and speed up inference.",
    )

    new_query = "How does model quantisation reduce memory?"
    print(f"\n  Query after adding fact: {new_query!r}")
    agent.answer(new_query)

    print("\n" + "=" * 70)
    print("  Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
