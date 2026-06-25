# Advanced Day 21 — Exercises

# Exercise 1 — Add diagonal movement to A*
# Modify the A* planner in lesson.py to support 8-connected movement.
# Diagonal cost = sqrt(2). Does the robot take shorter paths?
# Visualize the difference.
# TODO

# Exercise 2 — Obstacle avoidance during navigation
# Add a "dynamic obstacle" that appears after the robot starts moving.
# When the robot reaches a waypoint, check if any new obstacles appeared.
# If the current path is blocked, re-plan with A*.
# Simulate: after visiting beacon 2, add a new wall blocking the path to beacon 3.
# TODO

# Exercise 3 — Battery-aware mission planning
# Add battery to the robot: starts at 100, drains 1/step.
# If battery < 25, robot must return to CHARGER at (10,10) before continuing.
# Modify MissionFSM to handle the RECHARGE state.
# Plot battery level over time.
# TODO

# Exercise 4 — Multi-robot coordination
# Add a second robot (Robot B) starting at (18,18).
# Both robots share the same mission order but split it:
#   Robot A takes beacons 1,3,5
#   Robot B takes beacons 2,4
# Ensure they don't plan paths through each other's current position.
# Visualize both robots' trajectories.
# TODO

# Exercise 5 — Mission report generation
# After the mission, generate a text report:
#   - Mission start time, end time, total duration
#   - Beacons visited (in order) with arrival times
#   - Total distance traveled (sum of Euclidean distances)
#   - Average speed
#   - Any beacons that were skipped and why
# Save report to mission_report.txt
# TODO
