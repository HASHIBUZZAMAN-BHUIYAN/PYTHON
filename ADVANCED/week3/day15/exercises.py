# Advanced Day 15 — Exercises

import numpy as np, matplotlib.pyplot as plt

L1, L2, L3 = 1.0, 0.8, 0.5

# Exercise 1 — 3D rotation matrices
# Implement Rx(angle), Ry(angle), Rz(angle) — 3×3 rotation matrices.
# Verify that Rz(90°) rotates [1,0,0] to [0,1,0].
# Compose Rz(45°) @ Ry(30°) and apply to [1,0,0].
# TODO

# Exercise 2 — 3-link arm forward kinematics
# Extend FK to a 3-link arm (L1, L2, L3).
# origin → j1 → j2 → end-effector
# FK: angles θ1, θ2, θ3 (all relative to previous link)
# Compute positions for θ1=30°, θ2=45°, θ3=-30°.
# Visualize the arm.
# TODO

# Exercise 3 — Differential kinematics (Jacobian)
# The Jacobian maps joint velocities to end-effector velocity: v = J * dθ
# For 2-link arm:
# J = [[-L1*sin(θ1) - L2*sin(θ1+θ2),  -L2*sin(θ1+θ2)],
#       [L1*cos(θ1) + L2*cos(θ1+θ2),   L2*cos(θ1+θ2)]]
# Given dθ1=0.1 rad/s, dθ2=0.2 rad/s at θ1=π/4, θ2=π/6,
# compute end-effector velocity vector.
# TODO

# Exercise 4 — Differential-drive robot
# A differential-drive robot has two wheels (left, right).
# v = (v_r + v_l) / 2   (linear speed)
# ω = (v_r - v_l) / d   (angular speed, d = wheel base)
# Simulate the robot for 5 seconds with:
# t < 2: v_l=1.0, v_r=1.0 (straight)
# 2 ≤ t < 3: v_l=0.5, v_r=1.0 (turn right)
# t ≥ 3: v_l=1.0, v_r=1.0 (straight again)
# dt=0.05, d=0.3. Plot the path (x,y).
# TODO

# Exercise 5 — Workspace with obstacle avoidance
# Plot the 2-link arm workspace (from lesson.py).
# Add a circular obstacle at (0.8, 0.5) with radius 0.2.
# Color reachable points green (no collision) and red (inside obstacle).
# TODO
