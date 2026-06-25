# Advanced Day 40 — Exercises

# Exercise 1 — Custom eval metric
# Implement a BLEU-1 (unigram overlap) evaluation metric.
# For each response, compute: |reference_words ∩ hypothesis_words| / |hypothesis_words|
# Run it on 5 test cases and compare to the simple "contains" metric.
# TODO

# Exercise 2 — Adversarial input testing
# Create 10 adversarial prompts designed to bypass guardrails.
# Examples: Unicode lookalikes, spacing tricks, synonyms for blocked words.
# Try them against check_input(). How many bypass it?
# Improve the guardrail to catch more adversarial inputs.
# TODO

# Exercise 3 — Response schema validator
# Define a schema: {"answer": str, "confidence": float (0-1), "source": str}
# Validate that a JSON-structured agent response matches the schema.
# Try parsing 5 responses, catch malformed ones, and report errors.
# TODO

# Exercise 4 — Latency benchmark
# Time your mock_agent on 100 calls.
# Compute: mean, median, p95, p99 latency.
# Plot a histogram of latencies.
# TODO

# Exercise 5 — Reliability score
# Run the agent 20 times on the same input.
# Compute the "consistency score": fraction of runs with the same answer.
# Add gaussian noise to the selection (randomly pick a slightly different answer).
# Report consistency across all 5 eval cases.
# TODO
