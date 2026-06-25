# Advanced Day 15 Mini-Project — Animated 2-Link Robot Arm
# Follows a figure-8 trajectory using IK.
# ~50 MB RAM, <5s on CPU

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

L1, L2 = 1.0, 0.8

def fk(t1, t2):
    o  = np.array([0., 0.])
    j1 = np.array([L1*np.cos(t1), L1*np.sin(t1)])
    ee = j1 + np.array([L2*np.cos(t1+t2), L2*np.sin(t1+t2)])
    return o, j1, ee

def ik(x, y, elbow_up=True):
    cos_t2 = np.clip((x**2+y**2-L1**2-L2**2)/(2*L1*L2), -1, 1)
    sign   = 1 if elbow_up else -1
    t2     = np.arctan2(sign*np.sqrt(1-cos_t2**2), cos_t2)
    t1     = np.arctan2(y, x) - np.arctan2(L2*np.sin(t2), L1+L2*np.cos(t2))
    return t1, t2

# Figure-8 path
n_frames = 80
t_vals   = np.linspace(0, 2*np.pi, n_frames)
path_x   = 0.8 * np.sin(t_vals)
path_y   = 0.4 * np.sin(2 * t_vals)

# Compute IK for each point
waypoints = []
for px, py in zip(path_x, path_y):
    dist = np.sqrt(px**2 + py**2)
    if dist <= L1 + L2 - 0.05 and dist >= abs(L1 - L2) + 0.05:
        t1, t2 = ik(px, py)
        waypoints.append((t1, t2))
    else:
        waypoints.append(waypoints[-1] if waypoints else (0, 0))

# ─── Static plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot trajectory
axes[0].plot(path_x, path_y, "b--", alpha=0.4, label="Target path")
axes[0].set_xlim(-2, 2); axes[0].set_ylim(-1.5, 1.5)
axes[0].set_aspect("equal"); axes[0].grid(True, alpha=0.3)
axes[0].set_title("Figure-8 Trajectory (target)")

# Draw arm at a few waypoints
colors = plt.cm.viridis(np.linspace(0, 1, 10))
for i, (t1, t2) in enumerate(waypoints[::8]):
    o, j1, ee = fk(t1, t2)
    c = colors[i]
    axes[0].plot([o[0],j1[0],ee[0]], [o[1],j1[1],ee[1]], "o-", color=c, alpha=0.6, linewidth=2)

# Joint angle plot
t1s = [wp[0] for wp in waypoints]
t2s = [wp[1] for wp in waypoints]
axes[1].plot(np.degrees(t1s), label="θ1")
axes[1].plot(np.degrees(t2s), label="θ2")
axes[1].set_title("Joint Angles over Trajectory")
axes[1].set_xlabel("Waypoint"); axes[1].set_ylabel("Degrees"); axes[1].legend()

plt.tight_layout(); plt.savefig("robot_traj.png", dpi=80)
print("Saved robot_traj.png")

# ─── Animation ────────────────────────────────────────────────────────────────
fig_anim, ax_anim = plt.subplots(figsize=(6, 6))
ax_anim.set_xlim(-2.2, 2.2); ax_anim.set_ylim(-2.2, 2.2)
ax_anim.set_aspect("equal"); ax_anim.grid(True, alpha=0.3)
ax_anim.set_title("Robot Arm Following Figure-8")
ax_anim.plot(path_x, path_y, "b--", alpha=0.3, label="Path")

link1_line, = ax_anim.plot([], [], "o-", color="steelblue", linewidth=4, markersize=10)
link2_line, = ax_anim.plot([], [], "o-", color="orange",    linewidth=4, markersize=10)
ee_dot,     = ax_anim.plot([], [], "*",  color="gold",      markersize=15)
trace,      = ax_anim.plot([], [], "r-", alpha=0.3, linewidth=1)
trace_x, trace_y = [], []

def update(frame):
    t1, t2 = waypoints[frame % len(waypoints)]
    o, j1, ee = fk(t1, t2)
    link1_line.set_data([o[0],j1[0]], [o[1],j1[1]])
    link2_line.set_data([j1[0],ee[0]], [j1[1],ee[1]])
    ee_dot.set_data([ee[0]], [ee[1]])
    trace_x.append(ee[0]); trace_y.append(ee[1])
    trace.set_data(trace_x, trace_y)
    return link1_line, link2_line, ee_dot, trace

ani = animation.FuncAnimation(fig_anim, update, frames=len(waypoints),
                               interval=50, blit=True, repeat=True)
try:
    ani.save("robot_animation.gif", writer="pillow", fps=20)
    print("Saved robot_animation.gif")
except Exception as e:
    print(f"GIF save failed ({e}) — displaying live animation instead")
    plt.show()
plt.close()
