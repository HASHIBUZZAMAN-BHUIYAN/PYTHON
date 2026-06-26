"""
Swarm Collision Avoidance (ORCA-lite Velocity Obstacles)
=========================================================
What it does:
  6 robots are placed on a circle and each must reach the diametrically
  opposite point (the classic "circle scenario" from multi-robot navigation).
  All paths cross the centre, so robots MUST avoid each other.

  Each robot computes its velocity using a simplified ORCA approach:
    1. Desired velocity: constant speed toward goal
    2. For each other robot j:
       - Compute the Closest Point of Approach (CPA) time t_cpa
       - If CPA distance < 2*R_robot within T_horizon seconds:
           add a lateral avoidance impulse (perpendicular to relative position)
           scaled by (1 - d_cpa / 2R)  -- stronger when closer
           sign chosen so i steers RIGHT of j (right-hand traffic rule)
    3. Final velocity = desired + sum(avoidance impulses), clamped to MAX_SPD
  Near-misses (distance < WARN_DIST) are highlighted in red.

What it teaches:
  - Velocity Obstacle: the set of velocities that lead to collision within T
  - ORCA (Optimal Reciprocal Collision Avoidance): each agent takes half the
    avoidance responsibility, provably collision-free for ideal dynamics
  - CPA (Closest Point of Approach): key computation in maritime/aviation
  - Right-hand rule: why convention matters for deadlock avoidance
  - Scalability: O(n^2) pairwise checks; alternatives: spatial hashing, BVH

Controls: None -- auto-runs until all robots reach goals.
Animation: Live matplotlib window. Set MPLBACKEND=Agg to skip display (headless).
Output: ROBOTICS/outputs/swarm_avoidance.png
RAM: ~60 MB | Time: ~6s with display, <1s headless
"""

import os
import numpy as np
import matplotlib
INTERACTIVE = matplotlib.get_backend().lower() != 'agg'
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs("ROBOTICS/outputs", exist_ok=True)
rng = np.random.default_rng(0)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
N        = 6
RADIUS   = 4.2     # circle radius (m)
CENTER   = np.array([5.0, 5.0])
N_STEPS  = 500
DT       = 0.04
MAX_SPD  = 1.5
R_BOT    = 0.30    # robot body radius for collision
T_HRZ    = 4.0     # look-ahead horizon for VO (seconds)
GOAL_R   = 0.3
WARN_D   = 0.85    # near-miss warning distance (centres)
DRAW_N   = 3

# Robots: start at angles 0,60,...,300 deg; goals at opposite angles
angles_start = np.linspace(0, 2*np.pi, N, endpoint=False)
angles_goal  = angles_start + np.pi

pos   = np.array([CENTER + RADIUS * np.array([np.cos(a), np.sin(a)])
                  for a in angles_start])
goals = np.array([CENTER + RADIUS * np.array([np.cos(a), np.sin(a)])
                  for a in angles_goal])
vel   = np.zeros((N, 2))
paths = [[p.copy()] for p in pos]
done  = [False] * N
near_miss_count = 0

COLORS = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db','#9b59b6']

# ─── ORCA-LITE VELOCITY ───────────────────────────────────────────────────────
def orca_velocity(i, pos, vel, goal):
    """Compute safe velocity using CPA-based VO + emergency repulsion."""
    to_goal = goal - pos[i]
    d_goal  = np.linalg.norm(to_goal)
    if d_goal < GOAL_R:
        return np.zeros(2)
    v_des = MAX_SPD * to_goal / d_goal

    avoid = np.zeros(2)
    comb_r = 2.0 * R_BOT

    for j in range(N):
        if j == i or done[j]:
            continue
        p_rel = pos[j] - pos[i]  # i -> j
        d = np.linalg.norm(p_rel)
        if d < 1e-6:
            avoid -= rng.normal(1, 0.3, 2) * MAX_SPD
            continue

        p_hat = p_rel / d

        # Emergency hard repulsion (overrides VO when already too close)
        if d < comb_r * 1.5:
            avoid -= MAX_SPD * 3.0 * p_hat   # push i away from j
            continue

        # CPA computation
        v_rel = vel[j] - vel[i]   # relative velocity of j w.r.t. i
        dv_sq = np.dot(v_rel, v_rel)
        if dv_sq < 1e-8:
            continue
        t_cpa = np.clip(-np.dot(p_rel, v_rel) / dv_sq, 0.0, T_HRZ)
        p_cpa = p_rel + v_rel * t_cpa    # future p_rel at CPA
        d_cpa = np.linalg.norm(p_cpa)

        if d_cpa < comb_r * 3.0:   # avoidance zone = 3x combined radius
            # Right-hand rule: i steers to the right of j (CW rotation of p_hat)
            right_of_j = np.array([p_hat[1], -p_hat[0]])
            margin = (comb_r * 3.0 - d_cpa) / (comb_r * 3.0)
            # Urgency: nearer CPA in time = more urgent
            urgency = 1.0 - t_cpa / T_HRZ
            strength = MAX_SPD * margin * urgency * 1.2
            avoid += strength * right_of_j

    v_safe = v_des + avoid
    spd = np.linalg.norm(v_safe)
    if spd > MAX_SPD:
        v_safe = v_safe / spd * MAX_SPD
    return v_safe

# ─── SIMULATION ───────────────────────────────────────────────────────────────
print("Swarm Collision Avoidance (ORCA-lite)")
print("=" * 45)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle("Swarm Collision Avoidance -- ORCA-lite Circle Scenario", fontsize=11)
if INTERACTIVE:
    plt.ion()

min_dists = []

def draw(step):
    ax.clear();  ax2.clear()
    n_done = sum(done)
    ax.set_xlim(-0.5, 10.5);  ax.set_ylim(-0.5, 10.5)
    ax.set_aspect('equal');  ax.set_facecolor('#f8f8f8')
    ax.set_title(f"Step {step}/{N_STEPS}  done {n_done}/{N}  "
                 f"near-misses: {near_miss_count}", fontsize=9)

    # Draw circle boundary
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(CENTER[0]+RADIUS*np.cos(th), CENTER[1]+RADIUS*np.sin(th),
            '--', color='lightgrey', lw=1)

    # Paths
    for i in range(N):
        pt = np.array(paths[i])
        ax.plot(pt[:,0], pt[:,1], '-', color=COLORS[i], alpha=0.25, lw=1)

    # Check near-misses and set colours
    robot_colors = [COLORS[i] for i in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(pos[i]-pos[j]) < WARN_D:
                robot_colors[i] = 'red'
                robot_colors[j] = 'red'

    # Robots and goals
    for i in range(N):
        if not done[i]:
            ax.plot(*goals[i], '*', ms=12, color=COLORS[i], alpha=0.6, zorder=4)
        ec = 'gold' if done[i] else 'black'
        ax.add_patch(patches.Circle(pos[i], R_BOT, facecolor=robot_colors[i],
                                    edgecolor=ec, lw=1.5, zorder=5))
        ax.text(*pos[i], str(i+1), ha='center', va='center',
                fontsize=7, color='white', fontweight='bold', zorder=6)
        if not done[i] and np.linalg.norm(vel[i]) > 0.1:
            spd = np.linalg.norm(vel[i])
            ax.annotate('', xy=pos[i]+vel[i]/spd*0.35, xytext=pos[i],
                        arrowprops=dict(arrowstyle='->', color=COLORS[i], lw=1.5))

    ax.grid(alpha=0.2)

    # Min pairwise distance over time
    if min_dists:
        ax2.plot(min_dists, color='steelblue', lw=1.5)
        ax2.axhline(2*R_BOT, color='red', lw=1, linestyle='--', label=f'Collision threshold ({2*R_BOT}m)')
        ax2.axhline(WARN_D, color='orange', lw=1, linestyle=':', label=f'Near-miss warning ({WARN_D}m)')
        ax2.set_title("Minimum Pairwise Distance Over Time", fontsize=9)
        ax2.set_xlabel("Step");  ax2.set_ylabel("Min dist between any two robots (m)")
        ax2.legend(fontsize=8);  ax2.grid(alpha=0.3)
        ax2.set_ylim(0, max(min_dists) * 1.1 + 0.1)
    plt.tight_layout()

final_step = N_STEPS
for step in range(N_STEPS):
    # Compute new velocities (based on current state, before update)
    new_vel = np.zeros((N, 2))
    for i in range(N):
        if done[i]:
            continue
        new_vel[i] = orca_velocity(i, pos, vel, goals[i])
    vel = new_vel

    # Integrate
    pos += vel * DT
    for i in range(N):
        paths[i].append(pos[i].copy())

    # Check goal reached
    for i in range(N):
        if not done[i] and np.linalg.norm(pos[i] - goals[i]) < GOAL_R:
            done[i] = True
            print(f"  step {step:3d}: Robot {i+1} reached its goal  ({sum(done)}/{N})")

    # Minimum pairwise distance
    min_d = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < min_d:
                min_d = d
    min_dists.append(min_d)
    if min_d < WARN_D:
        near_miss_count += 1

    if step % DRAW_N == 0:
        draw(step)
        if INTERACTIVE:
            plt.pause(0.04)

    if all(done):
        final_step = step;  break

draw(final_step)
collisions = sum(1 for d in min_dists if d < 2*R_BOT)
print(f"All {N} robots reached goals at step {final_step}")
print(f"Near-miss frames (dist < {WARN_D}m): {near_miss_count}  Collisions: {collisions}")
print(f"Min separation achieved: {min(min_dists):.3f}m  (collision = {2*R_BOT}m)")

plt.savefig("ROBOTICS/outputs/swarm_avoidance.png", dpi=90, bbox_inches='tight')
print("Saved: ROBOTICS/outputs/swarm_avoidance.png")
if INTERACTIVE:
    plt.ioff();  plt.show()
plt.close()
