# ~120 MB RAM, ~2s on CPU
"""
Day 36 — Agent Memory: Exercises
=================================
Five practice exercises.  Each is a comment block with a TODO.
Do NOT look at solutions.py until you have attempted the exercise.
"""

# ──────────────────────────────────────────────
# Exercise 1 — Conversation Buffer with Overflow
# ──────────────────────────────────────────────
# TODO: Implement a ConversationBuffer class with max_turns=4.
#       The buffer should store {"role": "user"|"assistant", "content": str} dicts
#       (OpenAI-style message list).
#       When it overflows, the OLDEST user+assistant pair (2 messages) is dropped
#       together so the buffer never has an unpaired message.
#       After adding 6 turns, assert len(buffer) == 4.
#       Print all messages in order after the 6th turn is added.


# ──────────────────────────────────────────────
# Exercise 2 — Key-Value Store with Expiry
# ──────────────────────────────────────────────
# TODO: Build a KeyValueMemory class that stores (value, timestamp) pairs.
#       Add a `ttl` parameter (time-to-live in seconds, default 60).
#       `recall(key)` should return None and remove the key if it has expired.
#       Demonstrate: store a key with ttl=1, sleep 2 seconds, recall it → None.
#       Also store a key with ttl=60, recall immediately → value returned.
#       Use time.time() for timestamps.


# ──────────────────────────────────────────────
# Exercise 3 — TF-IDF Memory Search
# ──────────────────────────────────────────────
# TODO: Create a TFIDFMemory class (use sklearn TfidfVectorizer + cosine_similarity).
#       Load 10 arbitrary factual sentences of your choice as the knowledge base.
#       Write a search(query, top_k=2) method that returns a list of
#       (sentence, similarity_score) tuples, highest similarity first.
#       Run 3 test queries and print results.
#       If sklearn is not installed, fall back to a simple word-overlap Jaccard score.


# ──────────────────────────────────────────────
# Exercise 4 — Memory Consolidation Agent
# ──────────────────────────────────────────────
# TODO: Build a ConsolidatingAgent that uses a 3-turn buffer.
#       When the buffer overflows, the agent "summarises" the evicted turn
#       into a long-term dict using this rule:
#         key   = first 3 words of the user message (lowercase, joined by "_")
#         value = assistant message truncated to 50 characters
#       Simulate 8 conversation turns (hardcoded user/assistant pairs).
#       After all turns, print the long-term store (should have 5 entries).


# ──────────────────────────────────────────────
# Exercise 5 — Multi-Type Memory Manager
# ──────────────────────────────────────────────
# TODO: Implement a MemoryManager class that holds:
#         - A ConversationBuffer (max 5 turns)
#         - A KeyValueMemory (persistent JSON dict, in-memory is fine)
#         - A TFIDFMemory (or word-overlap fallback)
#       Add a unified `remember(user, assistant)` method that:
#         1. Adds the turn to the ConversationBuffer
#         2. Splits the assistant response into sentences and adds each to TFIDFMemory
#       Add a unified `retrieve(query)` method that:
#         1. Searches TFIDFMemory (returns best match)
#         2. Falls back to KeyValueMemory lookup if TF-IDF score < 0.1
#       Run 5 turns through remember(), then run 3 queries through retrieve().
#       Print what was found and from which store.
