# Advanced Day 16 — Solutions
import numpy as np
import matplotlib.pyplot as plt

# ─── shared PID class ──────────────────────────────────────────────────────────
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0., dt=0.01,
                 out_min=-float("inf"), out_max=float("inf"), anti_windup=False):
        self.Kp=Kp; self.Ki=Ki; self.Kd=Kd
        self.setpoint=setpoint; self.dt=dt
        self.out_min=out_min; self.out_max=out_max
        self.anti_windup=anti_windup
        self.integral=0.; self.prev_error=0.

    def step(self, measurement):
        error = self.setpoint - measurement
        P = self.Kp * error
        D = self.Kd * (error - self.prev_error) / self.dt
        self.prev_error = error
        raw = P + self.Ki * self.integral + D
        output = np.clip(raw, self.out_min, self.out_max)
        # Only integrate when not saturated (anti-windup)
        if self.anti_windup:
            if output == raw:
                self.integral += error * self.dt
        else:
            self.integral += error * self.dt
        return output

    def reset(self):
        self.integral=0.; self.prev_error=0.

# ─── Exercise 1 — mass-spring-damper PID tuning ────────────────────────────────
print("=== Exercise 1: Mass-Spring-Damper ===")
m, c, k = 1.0, 0.5, 2.0
dt, n = 0.01, 800
setpoint = 1.0

pid = PIDController(Kp=8.0, Ki=5.0, Kd=3.0, setpoint=setpoint, dt=dt,
                    out_min=-50., out_max=50.)
x, xdot = 0., 0.
times, xs = [], []
for i in range(n):
    F = pid.step(x)
    xddot = (F - c*xdot - k*x) / m
    xdot += xddot * dt
    x    += xdot * dt
    times.append(i*dt); xs.append(x)

ss_err   = abs(setpoint - xs[-1])
overshoot = max(0., (max(xs) - setpoint) / setpoint * 100)
print(f"  Final x={xs[-1]:.4f}  Overshoot={overshoot:.1f}%  SS-error={ss_err:.4f}")

# ─── Exercise 2 — Anti-windup comparison ───────────────────────────────────────
print("\n=== Exercise 2: Anti-Windup ===")
def run_saturated(anti_windup, label):
    pid2 = PIDController(Kp=3., Ki=2., Kd=0.5, setpoint=1., dt=0.01,
                         out_min=-10., out_max=10., anti_windup=anti_windup)
    x2, xdot2 = 0., 0.
    out = []
    for _ in range(800):
        F = pid2.step(x2)
        xddot = (F - 0.5*xdot2 - 2.*x2) / 1.
        xdot2 += xddot * 0.01; x2 += xdot2 * 0.01
        out.append(x2)
    overshoot = max(0., (max(out)-1.)/1.*100)
    print(f"  {label}: overshoot={overshoot:.1f}%  final={out[-1]:.4f}")
    return out

out_no_aw = run_saturated(False, "No anti-windup ")
out_aw    = run_saturated(True,  "With anti-windup")

fig, ax = plt.subplots(figsize=(9,4))
ax.plot(out_no_aw, label="No anti-windup")
ax.plot(out_aw,    label="With anti-windup")
ax.axhline(1., color="k", linestyle="--", label="Setpoint")
ax.set_title("Anti-Windup Comparison"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("anti_windup.png", dpi=80); plt.close()
print("  Saved anti_windup.png")

# ─── Exercise 3 — Joint PID controller ────────────────────────────────────────
print("\n=== Exercise 3: Robot Joint PID ===")
target_angle = np.pi/3
pid3 = PIDController(Kp=10., Ki=1., Kd=2., setpoint=target_angle, dt=0.01,
                     out_min=-20., out_max=20.)
angle, omega = 0., 0.
friction = 0.1
angles, controls = [], []
for _ in range(500):
    u = pid3.step(angle)
    omega += (u - friction*omega) * 0.01
    angle += omega * 0.01
    angles.append(np.degrees(angle)); controls.append(u)

fig, axes = plt.subplots(2,1, figsize=(9,6), sharex=True)
axes[0].plot(angles); axes[0].axhline(np.degrees(target_angle), color="r", linestyle="--", label="Target")
axes[0].set_ylabel("Angle (°)"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(controls, color="orange"); axes[1].set_ylabel("Control signal"); axes[1].grid(alpha=0.3)
axes[1].set_xlabel("Step")
plt.suptitle("Joint 1 PID Control"); plt.tight_layout()
plt.savefig("joint_pid.png", dpi=80); plt.close()
print(f"  Final angle={angles[-1]:.2f}°  Target={np.degrees(target_angle):.2f}°")
print("  Saved joint_pid.png")

# ─── Exercise 4 — Step response metrics ───────────────────────────────────────
print("\n=== Exercise 4: Step Response Metrics ===")
arr = np.array(xs)
sp  = setpoint
# Rise time: first time crosses 10% → 90%
rise10 = np.argmax(arr >= 0.1*sp) * dt
rise90 = np.argmax(arr >= 0.9*sp) * dt
rise_time = rise90 - rise10
# Settling time: last time outside ±5% band
band = 0.05 * sp
outside = np.where(np.abs(arr - sp) > band)[0]
settle_time = outside[-1] * dt if len(outside) else 0.
print(f"  Rise time    = {rise_time:.3f}s")
print(f"  Settling time= {settle_time:.3f}s")
print(f"  Overshoot    = {overshoot:.1f}%")

# ─── Exercise 5 — Cascade PID ──────────────────────────────────────────────────
print("\n=== Exercise 5: Cascade PID ===")
outer = PIDController(Kp=2.,  Ki=0.5, Kd=0.1, setpoint=1., dt=0.01, out_min=-3., out_max=3.)
inner = PIDController(Kp=5.,  Ki=1.,  Kd=0.2, setpoint=0., dt=0.01, out_min=-20., out_max=20.)
pos, vel = 0., 0.
positions, velocities = [], []
for _ in range(800):
    vel_sp   = outer.step(pos)
    inner.setpoint = vel_sp
    force    = inner.step(vel)
    vel     += force * 0.01 - 0.2 * vel * 0.01  # simple drag
    pos     += vel  * 0.01
    positions.append(pos); velocities.append(vel)

fig, axes = plt.subplots(2,1, figsize=(9,6), sharex=True)
axes[0].plot(positions); axes[0].axhline(1., color="r", linestyle="--", label="Setpoint")
axes[0].set_ylabel("Position"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(velocities, color="green"); axes[1].set_ylabel("Velocity"); axes[1].grid(alpha=0.3)
axes[1].set_xlabel("Step")
plt.suptitle("Cascade PID (position → velocity)"); plt.tight_layout()
plt.savefig("cascade_pid.png", dpi=80); plt.close()
print(f"  Final position={positions[-1]:.4f}")
print("  Saved cascade_pid.png")
