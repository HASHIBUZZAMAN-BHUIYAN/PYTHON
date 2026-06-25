# Advanced Day 16 — Exercises
import numpy as np, matplotlib.pyplot as plt

# Exercise 1 — Tune a PID for a second-order system
# System: mass-spring-damper
# m*x'' + c*x' + k*x = F(t)
# State: [x, x'] updated with Euler integration.
# m=1, c=0.5, k=2, dt=0.01, target x=1.0
# Start at x=0, x'=0.
# Tune Kp, Ki, Kd to reach target in <5s with <10% overshoot.
# TODO

# Exercise 2 — Anti-windup
# Integrator windup occurs when the system saturates (output is clamped)
# but the integral keeps growing.
# Implement anti-windup: only accumulate integral when NOT saturated.
# Test: simulate system with saturation at ±10. Compare with and without anti-windup.
# TODO

# Exercise 3 — PID position controller for robot arm joint
# Use the 2-link arm from Day 15.
# Add a PID controller to drive joint 1 to a target angle (e.g. π/3).
# System model: angle velocity += (PID_output - friction * omega) * dt
# Friction = 0.1, initial angle = 0.
# Plot: target angle, actual angle, control signal.
# TODO

# Exercise 4 — Step response analysis
# For your best-tuned PID from Exercise 1, compute:
# - Rise time (time to go from 10% to 90% of setpoint)
# - Settling time (time to stay within 5% of setpoint)
# - Overshoot (percentage)
# Print these metrics.
# TODO

# Exercise 5 — Cascade PID (inner/outer loop)
# Outer loop: position controller (setpoint = 1.0 m)
# Inner loop: velocity controller
# Outer PID output = velocity setpoint
# Inner PID output = force/control
# Implement both PIDs interacting. Plot position and velocity over time.
# TODO
