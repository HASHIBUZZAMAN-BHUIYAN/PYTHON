# Advanced Day 16 — PID Controller
# ~30 MB RAM, <2s on CPU

import numpy as np
import matplotlib.pyplot as plt

# ─── 1. PID CONCEPT ───────────────────────────────────────────────────────────
print("""
=== PID Controller ===

A PID controller computes a control signal u(t) to drive a system
toward a target setpoint.

u(t) = Kp * e(t)  +  Ki * ∫e(t)dt  +  Kd * de(t)/dt

where e(t) = setpoint - measured_value

  P (Proportional): reacts to current error  → faster response
  I (Integral):     reacts to accumulated error → eliminates steady-state error
  D (Derivative):   reacts to rate of change  → dampens oscillation

Tuning: increase Kp until oscillation, add Kd to dampen, add Ki to remove offset.
""")

# ─── 2. PID CLASS ─────────────────────────────────────────────────────────────
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0., dt=0.01,
                 output_min=-float("inf"), output_max=float("inf")):
        self.Kp = Kp; self.Ki = Ki; self.Kd = Kd
        self.setpoint   = setpoint
        self.dt         = dt
        self.integral   = 0.
        self.prev_error = 0.
        self.output_min = output_min
        self.output_max = output_max

    def step(self, measurement):
        error      = self.setpoint - measurement
        P          = self.Kp * error
        self.integral += error * self.dt
        I          = self.Ki * self.integral
        D          = self.Kd * (error - self.prev_error) / self.dt
        self.prev_error = error
        output = np.clip(P + I + D, self.output_min, self.output_max)
        return output

    def reset(self):
        self.integral = 0.; self.prev_error = 0.

# ─── 3. SIMULATE A SIMPLE SYSTEM ─────────────────────────────────────────────
print("=== 3. First-Order System Simulation ===")
print("""
System: a "room temperature" model.
  T(t+dt) = T(t) + (u(t) - 0.5*(T(t)-T_outside)) * dt
  u(t) = heater power (control signal)
""")

def simulate(Kp, Ki, Kd, setpoint=22., T_init=15., T_outside=5., dt=0.1, n_steps=300, label=""):
    """Simulate temperature control."""
    pid = PIDController(Kp, Ki, Kd, setpoint, dt, output_min=0., output_max=50.)
    T = T_init
    temps, controls, times = [], [], []
    for i in range(n_steps):
        u = pid.step(T)
        T = T + (u - 0.5*(T - T_outside)) * dt
        temps.append(T)
        controls.append(u)
        times.append(i * dt)
    ss_error = abs(setpoint - temps[-1])
    overshoot = max(0, max(temps) - setpoint)
    print(f"{label:<30}: final_T={temps[-1]:.2f}°C  overshoot={overshoot:.2f}  ss_error={ss_error:.3f}")
    return times, temps, controls

configs = [
    (2.0, 0.0, 0.0, "P only"),
    (2.0, 0.5, 0.0, "PI"),
    (2.0, 0.5, 1.0, "PID (tuned)"),
    (5.0, 0.5, 0.1, "PID (aggressive Kp)"),
]

all_results = {}
for Kp, Ki, Kd, label in configs:
    times, temps, controls = simulate(Kp, Ki, Kd, label=label)
    all_results[label] = (times, temps, controls)

# ─── 4. VISUALIZE ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for label, (times, temps, controls) in all_results.items():
    axes[0].plot(times, temps, label=label, linewidth=1.5)
axes[0].axhline(22., color="black", linestyle="--", linewidth=1, label="Setpoint (22°C)")
axes[0].set_title("Temperature Control — PID Comparison")
axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Temperature (°C)")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

label = "PID (tuned)"
axes[1].plot(all_results[label][0], all_results[label][2], color="tomato")
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].set_title(f"Control Signal ({label})"); axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Heater power"); axes[1].grid(True, alpha=0.3)

plt.tight_layout(); plt.savefig("pid_control.png", dpi=80)
print("\nSaved pid_control.png")

# ─── 5. ROBOT PATH FOLLOWING WITH PID ────────────────────────────────────────
print("\n=== 5. Robot Line Following with PID ===")
print("""
A simple robot tries to follow a straight line (y=0).
Its heading θ deviates due to noisy motion.
A PID controller corrects the steering angle to bring y→0.
""")

dt = 0.05
pid_steer = PIDController(Kp=3.0, Ki=0.5, Kd=1.0, setpoint=0., dt=dt)
x, y, theta, v = 0., 0.5, -0.1, 1.0    # start offset from line
xs, ys = [x], [y]
np.random.seed(42)
for _ in range(200):
    steer  = pid_steer.step(y)          # y is "error" — deviation from y=0
    theta  += steer * dt + np.random.normal(0, 0.02)
    x      += v * np.cos(theta) * dt
    y      += v * np.sin(theta) * dt
    xs.append(x); ys.append(y)

fig2, ax = plt.subplots(figsize=(10, 3))
ax.plot(xs, ys, label="Robot path", color="steelblue")
ax.axhline(0., color="red", linestyle="--", linewidth=2, label="Target line (y=0)")
ax.set_title("PID Line Following"); ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.tight_layout(); plt.savefig("pid_line_follow.png", dpi=80)
print("Saved pid_line_follow.png")
plt.show()
