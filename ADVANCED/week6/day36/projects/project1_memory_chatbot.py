# ~120 MB RAM, ~2s on CPU
"""
Project 1 — Memory Chatbot
===========================
A rule-based chatbot with a 5-turn rolling conversation buffer.
- ConversationBuffer keeps the last 5 (user, assistant) pairs.
- A MockLLM produces deterministic responses based on keyword matching.
- Simulates a 10-turn conversation and prints the context window after each turn.
- No API key required.
"""

from collections import deque

# ──────────────────────────────────────────────
# MOCK LLM — deterministic keyword-based responses
# ──────────────────────────────────────────────

RESPONSE_TABLE = {
    "python":       "Python is a high-level, interpreted programming language known for its readability.",
    "list":         "A list in Python is an ordered, mutable sequence. Example: [1, 2, 3].",
    "dict":         "A dictionary maps keys to values. Example: {'name': 'Alice'}.",
    "function":     "A function is a reusable block of code defined with the `def` keyword.",
    "class":        "A class is a blueprint for creating objects, defined with the `class` keyword.",
    "loop":         "Python supports `for` and `while` loops for iteration.",
    "memory":       "Agents use short-term and long-term memory to maintain context across turns.",
    "agent":        "An agent perceives its environment and takes actions to achieve goals.",
    "machine learning": "Machine learning is a subset of AI where models learn from data.",
    "hello":        "Hello! I'm a Python-aware chatbot. Ask me anything about Python or AI.",
}

DEFAULT_RESPONSE = "That's an interesting question. Could you rephrase or give more detail?"


def mock_llm(user_message: str, context: list) -> str:
    """
    Deterministic mock LLM.
    Checks user_message for keywords; also checks last assistant message
    in context to avoid repeating the same answer.
    """
    msg_lower = user_message.lower()
    for keyword, response in RESPONSE_TABLE.items():
        if keyword in msg_lower:
            # Personalise slightly by mentioning prior context length
            if context:
                return f"{response} (I have {len(context)} prior turns in memory.)"
            return response
    return DEFAULT_RESPONSE


# ──────────────────────────────────────────────
# CONVERSATION BUFFER
# ──────────────────────────────────────────────

class ConversationBuffer:
    """Rolling window of the last max_turns conversation pairs."""

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self._pairs: deque = deque(maxlen=max_turns)

    def add(self, user: str, assistant: str) -> None:
        self._pairs.append({"user": user, "assistant": assistant})

    def get_context(self) -> list:
        return list(self._pairs)

    def window_str(self) -> str:
        if not self._pairs:
            return "  (empty)"
        lines = []
        for i, p in enumerate(self._pairs, 1):
            lines.append(f"  [{i}] U: {p['user'][:50]}")
            lines.append(f"       A: {p['assistant'][:60]}")
        return "\n".join(lines)

    def __len__(self):
        return len(self._pairs)


# ──────────────────────────────────────────────
# SIMULATED 10-TURN CONVERSATION
# ──────────────────────────────────────────────

USER_MESSAGES = [
    "Hello there!",
    "What is Python?",
    "Can you explain what a list is?",
    "What about a dictionary?",
    "How do you define a function?",
    "What is a class?",
    "Tell me about loops in Python.",
    "What is machine learning?",
    "How does agent memory work?",
    "What makes a good AI agent?",
]


def run_chatbot():
    buffer = ConversationBuffer(max_turns=5)

    print("=" * 65)
    print("  MEMORY CHATBOT — 10-turn simulation (buffer size = 5)")
    print("=" * 65)

    for turn_num, user_msg in enumerate(USER_MESSAGES, 1):
        context = buffer.get_context()
        assistant_msg = mock_llm(user_msg, context)
        buffer.add(user_msg, assistant_msg)

        print(f"\n{'─' * 65}")
        print(f"  Turn {turn_num:>2}/10")
        print(f"  User     : {user_msg}")
        print(f"  Assistant: {assistant_msg[:80]}")
        print(f"\n  ── Context window after turn {turn_num} "
              f"({len(buffer)}/{buffer.max_turns} slots used) ──")
        print(buffer.window_str())

    print("\n" + "=" * 65)
    print("  Conversation complete.")
    print(f"  Final buffer holds turns {len(USER_MESSAGES) - 4}–{len(USER_MESSAGES)}")
    print("=" * 65)


if __name__ == "__main__":
    run_chatbot()
