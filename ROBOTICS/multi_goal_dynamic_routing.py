"""
Multi-Goal Dynamic Routing (Nearest-Neighbor TSP + Replanning)
===============================================================
What it does:
  A robot must visit all 6 goal locations in a 10x10 m arena.
  It uses two planning layers:
    1. GLOBAL PLAN (tour): Nearest-neighbor greedy heuristic selects the
       visit order -- at each step choose the closest unvisited goal
       (a classic approximation of the Travelling Salesman Problem)
    2. LOCAL NAV: Potential fields steers the robot toward the current
       sub-goal while dodging static obstacles
  At step DYN_STEP a new obstacle appears mid-arena, blocking the planned
  route. The planner immediately detects this, re-runs nearest-neighbor
  from the current position, and updates the tour in real time.

What it teaches:
  - Nearest-neighbor heuristic: O(n^2) TSP approximation
  - Why the NN tour can be poor (misses short cuts) vs. 2-opt improvements
  - Two-layer planning: global (where to go) vs. local (how to get there)
  - Dynamic replanning: why mobile robots need to update plans online
  - Potential fields: convenient local nav without a full grid search

Controls: None -- auto-runs.
Animation: Live matplotlib window. Set MPLBACKEND=Agg to skip display (headless).
Output: ROBOTICS/outputs/multi_goal_routing.png
RAM: ~60 MB | Time: ~8s with display, <1s headless
"""

import os
import numpy as np
import matplotlib
INTERACTIVE = matplotlib.get_backend().lower() != 'agg'
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs("ROBOTICS/outputs", exist_ok=True)
rng = np.random.default_rng(3)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
ARENA    = 10.0
N_STEPS  = 600
DT       = 0.04
MAX_SPD  = 1.5
K_ATT    = 4.0
K_REP    = 1.5
D_SAFE   = 1.0
GOAL_R   = 0.4
NOISE    = 0.2
DYN_STEP = 180    # dynamic obstacle appears at this step
DRAW_N   = 4

START = np.array([0.8, 0.8])

# Six goals spread around the arena
GOALS = np.array([
    [8.5, 1.5],
    [5.0, 8.5],
    [9.0, 7.5],
    [2.0, 5.5],
    [6.5, 4.0],
    [1.5, 9.0],
])

# Static obstacles
STATIC_OBS = [
    np.array([3.5, 2.5]),
    np.array([7.0, 3.0]),
    np.array([5.0, 6.0]),
    np.array([2.5, 8.0]),
]
OBS_R = 0.5

# ─── NEAREST-NEIGHBOR TOUR ────────────────────────────────────────────────────
def nn_tour(start_pos, unvisited_indices, goals):
    """Greedy nearest-neighbor tour from start_pos through unvisited goals."""
    order = []
    remaining = list(unvisited_indices)
    cur = start_pos.copy()
    while remaining:
        dists = [np.linalg.norm(goals[i] - cur) for i in remaining]
        best = remaining[np.argmin(dists)]
        order.append(best)
        cur = goals[best]
        remaining.remove(best)
    return order

# ─── POTENTIAL FIELDS ─────────────────────────────────────────────────────────
def nav_force(pos, target, obstacles):
    diff = target - pos
    d = np.linalg.norm(diff)
    fa = K_ATT * diff / max(d, 0.01)
    fr = np.zeros(2)
    for obs in obstacles:
        od = pos - obs
        dist = max(np.linalg.norm(od) - OBS_R, 0.01)
        if dist < D_SAFE:
            mag = K_REP * (1/dist - 1/D_SAFE) / dist**2
            fr += mag * od / (np.linalg.norm(od) + 1e-8)
    return fa + fr

# ─── SIMULATION STATE ─────────────────────────────────────────────────────────
print("Multi-Goal Dynamic Routing")
print("=" * 45)

pos          = START.copy()
path         = [pos.copy()]
visited      = []          # indices of visited goals
all_obs      = STATIC_OBS.copy()
dyn_placed   = False
tour       = nn_tour(pos, list(range(len(GOALS))), GOALS)
target_idx = tour[0] if tour else None
replanned    = False
escape_cd    = 0

fig, ax = plt.subplots(figsize=(8, 8))
if INTERACTIVE:
    plt.ion()

COLORS_GOAL = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e', '#9467bd', '#8c564b']

def draw(step):
    ax.clear()
    ax.set_xlim(-0.3, ARENA+0.3);  ax.set_ylim(-0.3, ARENA+0.3)
    ax.set_aspect('equal');  ax.set_facecolor('#f8f8f8')
    dyn_tag = " [DYN OBS APPEARED -- REPLANNING]" if (step == DYN_STEP and not replanned) else ""
    ax.set_title(f"Multi-Goal Routing  step {step}/{N_STEPS}  "
                 f"visited {len(visited)}/{len(GOALS)}{dyn_tag}", fontsize=9)

    for obs in STATIC_OBS:
        ax.add_patch(patches.Circle(obs, OBS_R, color='tomato', alpha=0.7, zorder=2))
    if dyn_placed:
        ax.add_patch(patches.Circle(all_obs[-1], OBS_R, facecolor='darkred',
                                    alpha=0.85, zorder=2, linewidth=2, linestyle='--',
                                    edgecolor='black', label='Dynamic obstacle'))

    for i, g in enumerate(GOALS):
        color = '#999' if i in visited else COLORS_GOAL[i % len(COLORS_GOAL)]
        marker = 'x' if i in visited else '*'
        ax.plot(g[0], g[1], marker, ms=16 if marker=='*' else 10,
                color=color, zorder=5)
        ax.text(g[0]+0.2, g[1]+0.2, str(i+1), fontsize=9, color=color, zorder=6)

    if tour:
        plan_pts = [pos.copy()] + [GOALS[i] for i in tour[len(visited):]]
        plan_arr = np.array(plan_pts)
        ax.plot(plan_arr[:,0], plan_arr[:,1], '--', color='orange',
                alpha=0.5, lw=1.5, zorder=3, label='Planned route')

    pts = np.array(path)
    ax.plot(pts[:,0], pts[:,1], '-', color='royalblue', alpha=0.4, lw=1, zorder=3)

    ax.plot(*START, 'ks', ms=8, zorder=5, label='Start')
    ax.plot(*pos,   'bo', ms=10, zorder=6, label='Robot')
    if dyn_placed:
        ax.legend(loc='upper right', fontsize=7)
    ax.grid(alpha=0.2)
    plt.tight_layout()

done = False
for step in range(N_STEPS):
    if step == DYN_STEP and not dyn_placed:
        dyn_obs = np.array([5.5, 3.5])
        all_obs.append(dyn_obs)
        dyn_placed = True
        remaining = [i for i in range(len(GOALS)) if i not in visited]
        tour = nn_tour(pos, remaining, GOALS)
        target_idx = tour[0] if tour else None
        replanned = True
        print(f"  step {step}: dynamic obstacle at {dyn_obs} -> replanning tour: "
              f"{[t+1 for t in tour]}")

    if target_idx is None:
        done = True; break

    noise_v = rng.normal(0, NOISE, 2)
    if escape_cd > 0:
        vel = K_ATT * (GOALS[target_idx] - pos) / max(np.linalg.norm(GOALS[target_idx]-pos), 0.1) + noise_v
        escape_cd -= 1
    else:
        vel = nav_force(pos, GOALS[target_idx], all_obs) + noise_v
        spd = np.linalg.norm(vel)
        if spd > MAX_SPD:
            vel = vel / spd * MAX_SPD
        if len(path) > 40:
            progress = np.linalg.norm(np.array(path[-1]) - np.array(path[-40]))
            if progress < 0.6:
                escape_cd = 30
                print(f"  step {step}: stuck approaching goal {target_idx+1} -> escape 30 steps")

    spd = np.linalg.norm(vel)
    if spd > MAX_SPD:
        vel = vel / spd * MAX_SPD

    pos = np.clip(pos + vel * DT, 0.0, ARENA)
    path.append(pos.copy())

    if np.linalg.norm(pos - GOALS[target_idx]) < GOAL_R:
        visited.append(target_idx)
        print(f"  step {step}: reached goal {target_idx+1}  ({len(visited)}/{len(GOALS)})")
        remaining = [i for i in range(len(GOALS)) if i not in visited]
        tour = nn_tour(pos, remaining, GOALS)
        target_idx = tour[0] if tour else None

    if step % DRAW_N == 0:
        draw(step)
        if INTERACTIVE:
            plt.pause(0.04)

draw(min(step, N_STEPS-1))
if len(visited) == len(GOALS):
    print(f"All {len(GOALS)} goals visited!")
else:
    print(f"Completed {len(visited)}/{len(GOALS)} goals at step {step}")

plt.savefig("ROBOTICS/outputs/multi_goal_routing.png", dpi=90, bbox_inches='tight')
print("Saved: ROBOTICS/outputs/multi_goal_routing.png")
if INTERACTIVE:
    plt.ioff();  plt.show()
plt.close()
