# Advanced Day 41 — Exercises

# Exercise 1 — A* pathfinding
# Implement A* search with Manhattan distance heuristic.
# Compare A* vs BFS: steps taken, nodes explored, path length.
# Run both on the same 8x8 grid with 10 random walls.
# TODO

# Exercise 2 — Sensor noise
# Add Gaussian noise to the agent's position sensor: pos = actual_pos + N(0, 0.5).
# Round to nearest integer. How does the reactive agent perform with noisy sensors?
# Add a Kalman-style moving average to smooth sensor readings.
# TODO

# Exercise 3 — Obstacle avoidance
# Add 3 "dynamic obstacles" that move one step per turn (randomly).
# Modify the reactive agent to check for obstacle positions in its neighbors.
# Measure how often the agent gets stuck vs reaches the goal over 50 runs.
# TODO

# Exercise 4 — Coverage agent
# Implement a coverage robot that tries to visit EVERY non-wall cell.
# Use a systematic boustrophedon (lawnmower) pattern.
# Measure coverage percentage after 50 steps on a 6x6 grid.
# TODO

# Exercise 5 — Multi-objective reward
# Modify the GridWorld reward function:
#   +10 for reaching goal, +5 per dirt cleaned, -2 per step, -20 per wall bump
# Run both reactive and BFS agents and compare total reward across 3 episodes.
# TODO
