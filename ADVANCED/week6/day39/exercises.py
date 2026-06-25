# Advanced Day 39 — Exercises

# Exercise 1 — Star topology
# Implement a StarOrchestrator where one central coordinator agent
# sends the task to N worker agents simultaneously and collects results.
# Test with 3 workers: each has a different "specialty" and returns partial results.
# Coordinator merges the partial results into a final answer.
# TODO

# Exercise 2 — Escalation chain
# Build an EscalationChain with 3 agents of increasing expertise.
# Agent 1 (junior) tries first. If confidence < 0.5, pass to Agent 2 (senior).
# If senior also uncertain, pass to Agent 3 (expert).
# Simulate confidence with random.random().
# TODO

# Exercise 3 — Collaborative writing pipeline
# Build a 4-stage pipeline:
#   1. Outliner    → creates section headers
#   2. Writer      → expands each section into a paragraph
#   3. Fact-Checker→ adds [VERIFIED] or [UNVERIFIED] tags
#   4. Editor      → removes [UNVERIFIED] sections and polishes
# TODO

# Exercise 4 — Voting ensemble
# Create 5 sentiment classifier agents each using a different strategy:
#   lexicon, punctuation, length, caps, exclamations
# For a given text, collect all 5 votes and use weighted voting.
# Assign weights based on each agent's known accuracy: [0.6, 0.5, 0.5, 0.4, 0.7]
# TODO

# Exercise 5 — Agent supervision
# Build a SupervisorAgent that monitors a pipeline.
# After each step it checks: result length > 20 chars AND "error" not in result.
# If quality check fails, it re-runs that agent (max 2 retries).
# TODO
