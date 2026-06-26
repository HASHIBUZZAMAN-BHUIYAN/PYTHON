# Advanced Day 17 — Exercises

# Exercise 1 — Diagonal moves
# Modify A* to support 8-connected movement (including diagonals).
# Diagonal step cost = sqrt(2).
# Compare path length and nodes expanded vs 4-connected A*.

# Exercise 2 — Weighted A* (epsilon-A*)
# Weighted A*: f(n) = g(n) + w*h(n), w > 1 (e.g. w=1.5)
# Finds suboptimal but faster paths.
# Run A* with w=1.0, 1.5, 2.0 and compare: path length vs nodes expanded.

# Exercise 3 — Dynamic obstacles
# Add a "moving obstacle" that changes position every 20 steps.
# Replan (re-run A*) whenever the obstacle moves.
# Count total replanning calls.

# Exercise 4 — Q-learning reward shaping
# Standard reward: -1/step, +100 goal.
# Try adding a distance-based shaping reward: +0.1 * (prev_dist - new_dist)
# where dist = Manhattan distance to goal.
# Does shaping help the agent learn faster? Plot both reward curves.

# Exercise 5 — Value iteration
# Instead of Q-learning, implement Value Iteration (model-based DP).
# Transition: deterministic (action succeeds).
# Compute V(s) for all cells and derive the greedy policy.
# Compare path with A* path — should they match.
