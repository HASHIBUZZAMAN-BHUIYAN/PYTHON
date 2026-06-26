"""
Smooth Trajectory Planning (Polynomial Interpolation)
======================================================
What it does:
  Plans motions for a 2D robot along a sequence of waypoints and compares
  three trajectory types side by side:
    1. LINEAR: constant-speed straight-line segments between waypoints
    2. CUBIC: cubic polynomial (3rd-order) -- matches position + zero-velocity
       at segment boundaries, giving smooth speed-up/slow-down profiles
    3. QUINTIC: quintic polynomial (5th-order) -- additionally matches zero
       acceleration at boundaries, giving smooth jerk profiles

  For each trajectory the animation shows:
    - The 2D path in the workspace (coloured by speed)
    - Velocity magnitude over time
    - Acceleration magnitude over time

  The vehicle (dot) is animated travelling along each trajectory simultaneously
  so the differences in motion feel are immediately visible.

What it teaches:
  - Why linear trajectories produce infinite acceleration at waypoints
  - Cubic polynomial derivation: 4 unknowns (a0..a3), 4 boundary conditions
  - Quintic polynomial: 6 unknowns, also constrains acceleration at endpoints
  - Jerk: rate of change of acceleration; smooth trajectories minimise jerk
  - Practical use in robot joint-space and Cartesian-space planning

Controls: None -- auto-runs.
Animation: Live matplotlib window. Set MPLBACKEND=Agg to skip display (headless).
Output: ROBOTICS/outputs/trajectory_planning.png
RAM: ~60 MB | Time: ~7s with display, <1s headless
"""

import os
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib
INTERACTIVE = matplotlib.get_backend().lower() != 'agg'
import matplotlib.pyplot as plt

os.makedirs("ROBOTICS/outputs", exist_ok=True)

# ─── WAYPOINTS ────────────────────────────────────────────────────────────────
# 5 waypoints forming a curved route through the workspace
WP = np.array([
    [0.0,  0.0],
    [2.5,  3.0],
    [5.5,  1.5],
    [7.0,  5.0],
    [9.5,  4.5],
])
N_WP = len(WP)
T_SEG   = 1.0    # time per segment (s)
T_TOTAL = (N_WP - 1) * T_SEG
N_FRAMES_PER_SEG = 60
N_FRAMES         = N_FRAMES_PER_SEG * (N_WP - 1)
DRAW_N = 2

# ─── TRAJECTORY GENERATORS ────────────────────────────────────────────────────
def linear_traj(waypoints, n_frames_per_seg):
    """Piecewise linear path, constant speed per segment."""
    pos, vel, acc = [], [], []
    dt = T_SEG / n_frames_per_seg
    for i in range(len(waypoints) - 1):
        p0, p1 = waypoints[i], waypoints[i+1]
        v = (p1 - p0) / T_SEG
        for _ in range(n_frames_per_seg):
            pos.append(p0 + v * dt * len(pos[i*n_frames_per_seg:]))
            vel.append(np.linalg.norm(v))
            acc.append(0.0)   # infinite impulse at waypoint, shown as 0 between
    return np.array(pos), np.array(vel), np.array(acc)

def cubic_poly_1d(p0, p1, v0, v1, T, t_arr):
    """Cubic polynomial with position + velocity boundary conditions."""
    # a0 + a1*t + a2*t^2 + a3*t^3
    # Boundary: p(0)=p0, p(T)=p1, p'(0)=v0, p'(T)=v1
    a0 = p0
    a1 = v0
    a2 = (3*(p1-p0)/T**2) - (2*v0+v1)/T
    a3 = (-2*(p1-p0)/T**3) + (v0+v1)/T**2
    p =  a0 + a1*t_arr + a2*t_arr**2 + a3*t_arr**3
    vd = a1 + 2*a2*t_arr + 3*a3*t_arr**2
    ad = 2*a2 + 6*a3*t_arr
    return p, vd, ad

def quintic_poly_1d(p0, p1, v0, v1, a0_bc, a1_bc, T, t_arr):
    """Quintic polynomial matching pos, vel, acc at both endpoints."""
    h = p1 - p0
    A = np.array([
        [T**3,   T**4,   T**5  ],
        [3*T**2, 4*T**3, 5*T**4],
        [6*T,    12*T**2,20*T**3],
    ])
    b = np.array([h - v0*T - 0.5*a0_bc*T**2,
                  v1 - v0 - a0_bc*T,
                  a1_bc - a0_bc])
    c3, c4, c5 = np.linalg.solve(A, b)
    c0, c1, c2 = p0, v0, 0.5*a0_bc
    p =  c0 + c1*t_arr + c2*t_arr**2 + c3*t_arr**3 + c4*t_arr**4 + c5*t_arr**5
    vd = c1 + 2*c2*t_arr + 3*c3*t_arr**2 + 4*c4*t_arr**3 + 5*c5*t_arr**4
    ad = 2*c2 + 6*c3*t_arr + 12*c4*t_arr**2 + 20*c5*t_arr**3
    return p, vd, ad

def build_cubic(waypoints, n_frames_per_seg):
    pos_x, pos_y, vel, acc = [], [], [], []
    t_arr = np.linspace(0, T_SEG, n_frames_per_seg, endpoint=False)
    for i in range(len(waypoints) - 1):
        x0, y0 = waypoints[i];   x1, y1 = waypoints[i+1]
        xp, xv, xa = cubic_poly_1d(x0, x1, 0.0, 0.0, T_SEG, t_arr)
        yp, yv, ya = cubic_poly_1d(y0, y1, 0.0, 0.0, T_SEG, t_arr)
        pos_x.extend(xp);  pos_y.extend(yp)
        vel.extend(np.sqrt(xv**2 + yv**2))
        acc.extend(np.sqrt(xa**2 + ya**2))
    return np.column_stack([pos_x, pos_y]), np.array(vel), np.array(acc)

def build_quintic(waypoints, n_frames_per_seg):
    pos_x, pos_y, vel, acc = [], [], [], []
    t_arr = np.linspace(0, T_SEG, n_frames_per_seg, endpoint=False)
    for i in range(len(waypoints) - 1):
        x0, y0 = waypoints[i];   x1, y1 = waypoints[i+1]
        xp, xv, xa = quintic_poly_1d(x0, x1, 0.0, 0.0, 0.0, 0.0, T_SEG, t_arr)
        yp, yv, ya = quintic_poly_1d(y0, y1, 0.0, 0.0, 0.0, 0.0, T_SEG, t_arr)
        pos_x.extend(xp);  pos_y.extend(yp)
        vel.extend(np.sqrt(xv**2 + yv**2))
        acc.extend(np.sqrt(xa**2 + ya**2))
    return np.column_stack([pos_x, pos_y]), np.array(vel), np.array(acc)

# ─── BUILD TRAJECTORIES ───────────────────────────────────────────────────────
print("Trajectory Planning (Linear vs Cubic vs Quintic)")
print("=" * 55)

lin_pos_list, lin_vel_list, lin_acc_list = [], [], []
dt = T_SEG / N_FRAMES_PER_SEG
for i in range(N_WP - 1):
    p0, p1 = WP[i], WP[i+1]
    v = (p1 - p0) / T_SEG
    spd = np.linalg.norm(v)
    for j in range(N_FRAMES_PER_SEG):
        lin_pos_list.append(p0 + v * j * dt)
        lin_vel_list.append(spd)
        lin_acc_list.append(0.0)
lin_pos = np.array(lin_pos_list)
lin_vel = np.array(lin_vel_list)
lin_acc = np.array(lin_acc_list)

cub_pos, cub_vel, cub_acc = build_cubic(WP, N_FRAMES_PER_SEG)
qui_pos, qui_vel, qui_acc = build_quintic(WP, N_FRAMES_PER_SEG)

print(f"Trajectory length: {N_FRAMES} frames  ({T_TOTAL:.1f}s)")
print(f"Max speed  linear={lin_vel.max():.2f}  cubic={cub_vel.max():.2f}  quintic={qui_vel.max():.2f}")
print(f"Max accel  linear={lin_acc.max():.2f}  cubic={cub_acc.max():.2f}  quintic={qui_acc.max():.2f}")

# ─── ANIMATION ────────────────────────────────────────────────────────────────
frames_t = np.arange(N_FRAMES) * (T_TOTAL / N_FRAMES)
COLS = {'linear': '#e74c3c', 'cubic': '#2ecc71', 'quintic': '#3498db'}

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
ax_path, ax_vel, ax_acc = axes
fig.suptitle("Trajectory Planning: Linear vs Cubic vs Quintic Polynomial", fontsize=11)
if INTERACTIVE:
    plt.ion()

def draw(frame):
    ax_path.clear();  ax_vel.clear();  ax_acc.clear()

    ax_path.set_xlim(-0.5, 10.5);  ax_path.set_ylim(-0.5, 6.0)
    ax_path.set_aspect('equal');   ax_path.set_facecolor('#f8f8f8')
    ax_path.set_title("2D Workspace (vehicles shown as dots)", fontsize=9)

    for pos, vel, col, lbl in [
            (lin_pos, lin_vel, COLS['linear'],  'Linear'),
            (cub_pos, cub_vel, COLS['cubic'],   'Cubic'),
            (qui_pos, qui_vel, COLS['quintic'], 'Quintic')]:
        ax_path.plot(pos[:frame+1, 0], pos[:frame+1, 1], '-', color=col, alpha=0.5, lw=1.5)
        ax_path.plot(*pos[frame], 'o', color=col, ms=10, label=lbl, zorder=5)

    ax_path.plot(WP[:, 0], WP[:, 1], 'k^', ms=8, zorder=4, label='Waypoints')
    ax_path.legend(fontsize=8, loc='upper left');  ax_path.grid(alpha=0.2)

    t = frames_t[:frame+1]
    ax_vel.plot(t, lin_vel[:frame+1], color=COLS['linear'],  lw=1.5, label='Linear')
    ax_vel.plot(t, cub_vel[:frame+1], color=COLS['cubic'],   lw=1.5, label='Cubic')
    ax_vel.plot(t, qui_vel[:frame+1], color=COLS['quintic'], lw=1.5, label='Quintic')
    ax_vel.set_title("Speed over Time", fontsize=9)
    ax_vel.set_xlabel("t (s)");  ax_vel.set_ylabel("Speed (m/s)")
    ax_vel.set_xlim(0, T_TOTAL);  ax_vel.legend(fontsize=8);  ax_vel.grid(alpha=0.3)

    ax_acc.plot(t, lin_acc[:frame+1], color=COLS['linear'],  lw=1.5, label='Linear')
    ax_acc.plot(t, cub_acc[:frame+1], color=COLS['cubic'],   lw=1.5, label='Cubic')
    ax_acc.plot(t, qui_acc[:frame+1], color=COLS['quintic'], lw=1.5, label='Quintic')
    ax_acc.set_title("Acceleration over Time", fontsize=9)
    ax_acc.set_xlabel("t (s)");  ax_acc.set_ylabel("Accel (m/s^2)")
    ax_acc.set_xlim(0, T_TOTAL);  ax_acc.legend(fontsize=8);  ax_acc.grid(alpha=0.3)
    plt.tight_layout()

for frame in range(N_FRAMES):
    if frame % DRAW_N == 0:
        draw(frame)
        if INTERACTIVE:
            plt.pause(0.03)

draw(N_FRAMES - 1)

plt.savefig("ROBOTICS/outputs/trajectory_planning.png", dpi=90, bbox_inches='tight')
print("Saved: ROBOTICS/outputs/trajectory_planning.png")
if INTERACTIVE:
    plt.ioff();  plt.show()
plt.close()
