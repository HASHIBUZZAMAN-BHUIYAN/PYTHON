# Advanced Day 16 Mini-Project — Autonomous Robot on a Curved Track
# PID-controlled differential-drive robot follows a sine-wave path.
# ~25 MB RAM, <2s on CPU

import numpy as np
import matplotlib.pyplot as plt

print("=== Day 16 Mini-Project: Curved Track Following with PID ===\n")

# ─── PID controller ─────────────────────────────────────────────────────────
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0., dt=0.05,
                 out_min=-float("inf"), out_max=float("inf")):
        self.Kp=Kp; self.Ki=Ki; self.Kd=Kd
        self.setpoint=setpoint; self.dt=dt
        self.out_min=out_min; self.out_max=out_max
        self.integral=0.; self.prev_error=0.

    def step(self, measurement):
        error = self.setpoint - measurement
        P = self.Kp * error
        self.integral = np.clip(self.integral + error*self.dt, -10., 10.)  # windup clamp
        I = self.Ki * self.integral
        D = self.Kd * (error - self.prev_error) / self.dt
        self.prev_error = error
        return np.clip(P + I + D, self.out_min, self.out_max)

# ─── Target track: sine wave ──────────────────────────────────────────────────
def target_y(x):
    return 0.5 * np.sin(0.5 * x)

def target_dy(x):
    return 0.25 * np.cos(0.5 * x)

# ─── Robot simulation ─────────────────────────────────────────────────────────
dt = 0.05
v  = 0.8      # constant forward speed
n_steps = 400

pid = PIDController(Kp=4.0, Ki=0.3, Kd=1.5, dt=dt, out_min=-3., out_max=3.)

x, y, theta = 0., 0.1, 0.   # slight initial offset
robot_xs, robot_ys = [x], [y]
errors, controls = [], []
np.random.seed(7)

for _ in range(n_steps):
    # Cross-track error: perpendicular distance to track
    ty = target_y(x)
    cte = ty - y  # positive = robot is below track

    # Desired heading = atan of track slope at current x
    desired_theta = np.arctan(target_dy(x))
    heading_error = desired_theta - theta

    # Blend CTE and heading error for smoother following
    composite_error = cte + 0.4 * heading_error

    steer = pid.step(-composite_error)  # negate: positive CTE → steer up (positive)

    theta += steer * dt + np.random.normal(0, 0.01)   # small noise
    x     += v * np.cos(theta) * dt
    y     += v * np.sin(theta) * dt

    robot_xs.append(x); robot_ys.append(y)
    errors.append(cte); controls.append(steer)

# ─── Metrics ─────────────────────────────────────────────────────────────────
cte_arr = np.array(errors)
print(f"Cross-track error  — mean |CTE|: {np.mean(np.abs(cte_arr)):.4f} m")
print(f"                  — max  |CTE|: {np.max(np.abs(cte_arr)):.4f} m")
print(f"                  — RMS   CTE : {np.sqrt(np.mean(cte_arr**2)):.4f} m")

# ─── Reference track for plotting ─────────────────────────────────────────────
x_ref = np.linspace(0., max(robot_xs)+0.5, 300)
y_ref = target_y(x_ref)

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(x_ref, y_ref, "r--", linewidth=2, label="Target track")
axes[0].plot(robot_xs, robot_ys, "b-", linewidth=1.2, label="Robot path", alpha=0.8)
axes[0].plot(robot_xs[0], robot_ys[0], "g^", markersize=10, label="Start")
axes[0].plot(robot_xs[-1], robot_ys[-1], "r*", markersize=12, label="End")
axes[0].set_title("Curved Track Following")
axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

t_arr = np.arange(len(errors)) * dt
axes[1].plot(t_arr, errors, color="steelblue")
axes[1].axhline(0., color="k", linestyle="--")
axes[1].fill_between(t_arr, errors, alpha=0.2, color="steelblue")
axes[1].set_title("Cross-Track Error vs Time")
axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("CTE (m)")
axes[1].grid(alpha=0.3)

axes[2].plot(t_arr, controls, color="tomato")
axes[2].axhline(0., color="k", linestyle="--")
axes[2].set_title("Steering Signal vs Time")
axes[2].set_xlabel("Time (s)"); axes[2].set_ylabel("Steering (rad/s)")
axes[2].grid(alpha=0.3)

plt.suptitle("Day 16 Mini-Project — PID Curved Track Following", fontsize=12)
plt.tight_layout(); plt.savefig("curved_track_pid.png", dpi=90)
print("\nSaved curved_track_pid.png")
plt.show()
