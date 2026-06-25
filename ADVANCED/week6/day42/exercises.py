# Advanced Day 42 — Final Exercises

# Exercise 1 — Add slot filling
# The intent classifier returns "calculation" with numbers and ops.
# Add slot_filling() that also extracts the intent's required arguments.
# For "what is the square root of 81?" → slots={"operation":"sqrt","number":81}
# For "what is 15% of 200?" → slots={"operation":"percent","a":15,"b":200}
# TODO

# Exercise 2 — Memory module
# Add a ConversationMemory class that stores the last N turns.
# It should expose:
#   remember(user_msg, assistant_msg) — stores a turn
#   context(n=3) — returns last n turns as a formatted string
#   relevant(query) — returns the 1 most relevant past turn (by keyword overlap)
# TODO

# Exercise 3 — Retry + fallback in the agent loop
# Modify agent_loop() to:
#   1. Try the primary tool (add/multiply/etc.)
#   2. On failure, fall back to "search" for the answer
#   3. On search failure, return a canned "I don't know" message
#   4. Track which fallback level was triggered
# TODO

# Exercise 4 — Evaluation harness
# Create a TestSuite with 10 query-answer pairs.
# Run the full agent loop and score accuracy.
# Also measure average response latency.
# Report pass/fail per case and overall accuracy.
# TODO

# Exercise 5 — End-to-end integration
# Combine:
#   - Intent classifier (lesson)
#   - Entity extractor (lesson)
#   - ToolRegistry (lesson)
#   - ConversationMemory (exercise 2)
#   - Safety guardrail from day40
# Build a simple REPL loop that:
#   1. Reads a query
#   2. Runs guardrail
#   3. Classifies intent
#   4. Calls the right tool
#   5. Saves turn to memory
#   6. Prints answer + memory context
# TODO
