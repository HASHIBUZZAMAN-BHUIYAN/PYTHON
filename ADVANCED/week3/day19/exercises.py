# Advanced Day 19 — Exercises

# Exercise 1 — Reflex agent with priority rules
# Build a reflex agent for a drone with these condition-action rules (in priority order):
#   1. If battery < 20%      → RETURN_TO_BASE
#   2. If obstacle_detected  → AVOID_OBSTACLE
#   3. If target_in_sight    → TRACK_TARGET
#   4. else                  → SEARCH_PATTERN
# Simulate 15 steps with random percepts and show state transitions.
# TODO

# Exercise 2 — FSM with timers
# Extend the SecurityRobotFSM to add a RECHARGE state.
# Battery drains 1 unit/step. When battery <= 20 → RECHARGE.
# RECHARGE fills battery at 5 units/step. When full (100) → PATROL.
# Show battery level column in the output.
# TODO

# Exercise 3 — Utility-based agent
# A robot can pick up objects A, B, C with utilities [10, 7, 3].
# Objects have positions on a 1D line: A=8, B=3, C=12. Robot starts at 0.
# Cost to move = 1/step. Net utility = object_utility - travel_distance.
# Build a utility agent that picks the highest net-utility object next.
# Print order of collection and total net utility.
# TODO

# Exercise 4 — Cooperative multi-agent
# Two vacuum agents work on a 4×4 grid.
# They must not step on each other's cell simultaneously.
# Agent 1 starts at (0,0), Agent 2 at (3,3).
# They alternate steps. If next cell is occupied by the other, WAIT.
# Run until all cells are clean. Count total steps for each agent.
# TODO

# Exercise 5 — Agent with memory (belief state)
# Environment: 5 doors (open or closed). Agent starts not knowing which are open.
# Each step: agent moves to a door and observes its state (observe = see true state).
# Agent maintains a belief dict {door: "open"/"closed"/"unknown"}.
# Print updated belief after each step until all doors are known.
# TODO
