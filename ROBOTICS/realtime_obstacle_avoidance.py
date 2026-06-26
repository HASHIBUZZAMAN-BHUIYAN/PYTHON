"""
Real-Time Obstacle Avoidance (Potential Fields)
================================================
What it does:
  Simulates a mobile robot navigating a 10x10 m arena with 6 circular
  obstacles using the Potential Fields algorithm:
    - Attractive force: constant-magnitude pull toward the goal (K_ATT)
    - Repulsive forces: push the robot away from obstacle surfaces (K_REP)
    - Local-minimum escape: robot tracks how far it has moved in the last
      STUCK_WIN steps; if progress < STUCK_THR, it ignores repulsion for
      ESCAPE_STEPS steps and drives straight to the goal

What it teaches:
  - Potential Fields: one of the oldest reactive motion planners (Khatib 1986)
  - Why local minima form (forces cancel, robot freezes or oscillates)
  - Position-progress based stuck detection vs. speed-based detection
  - Trade-off: repulsion strength vs. navigability through tight gaps
  - Simple escape heuristic vs. global re-planning with A* or RRT

Controls: None -- auto-runs until goal reached or N_STEPS elapsed.
Animation: Live matplotlib window. Set MPLBACKEND=Agg to skip display (headless).
Output: ROBOTICS/outputs/obstacle_avoidance.png
RAM: ~60 MB | Time: ~8s with display, <1s headless
"""

import os
import numpy as np
import matplotlib
INTERACTIVE = matplotlib.get_backend().lower() != 'agg'
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs("ROBOTICS/outputs", exist_ok=True)
rng = np.random.default_rng(13)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
ARENA      = 10.0
N_STEPS    = 500
DT         = 0.05
MAX_SPD    = 1.8
K_ATT      = 4.5    # goal attraction -- kept high so it dominates K_REP
K_REP      = 1.5    # repulsion gain
D_SAFE     = 1.0    # repulsion activates within this dist from obstacle surface
GOAL_R     = 0.35
NOISE      = 0.3    # Brownian noise prevents exact force-balance freezes
STUCK_WIN  = 35     # look-back window (steps) for progress check
STUCK_THR  = 0.8    # if moved < STUCK_THR m in STUCK_WIN steps -> escape
ESCAPE_N   = 40     # escape mode duration (steps -- ignore repulsion)
DRAW_N     = 4

START = np.array([0.8,  0.8])
GOAL  = np.array([9.2,  9.2])

# Obstacles placed BESIDE the main diagonal to keep a navigable channel
OBS = [
    (1.5, 4.5, 0.50),
    (3.5, 2.0, 0.50),
    (5.0, 4.2, 0.55),
    (7.0, 5.8, 0.50),
    (4.5, 7.5, 0.50),
    (8.0, 4.0, 0.50),
]

# ─── PHYSICS ──────────────────────────────────────────────────────────────────
def f_att(pos):
    diff = GOAL - pos
    d = np.linalg.norm(diff)
    return K_ATT * diff / max(d, 0.01)   # unit direction * gain

def f_rep(pos):
    total = np.zeros(2)
    for cx, cy, r in OBS:
        diff = pos - np.array([cx, cy])
        d = max(np.linalg.norm(diff) - r, 0.01)   # dist from surface
        if d < D_SAFE:
            mag = K_REP * (1.0/d - 1.0/D_SAFE) / d**2
            total += mag * diff / (np.linalg.norm(diff) + 1e-8)
    return total

# ─── SIMULATION ───────────────────────────────────────────────────────────────
print("Potential Fields Obstacle Avoidance")
print("=" * 45)

pos       = START.copy()
path      = [pos.copy()]
vel       = np.zeros(2)
done      = False
escape_cd = 0
escapes   = 0

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Potential Fields Obstacle Avoidance", fontsize=11)
if INTERACTIVE:
    plt.ion()

def draw(step, pos, vel, done_flag):
    ax.clear();  ax2.clear()
    ax.set_xlim(-0.3, ARENA+0.3);  ax.set_ylim(-0.3, ARENA+0.3)
    ax.set_aspect('equal');  ax.set_facecolor('#f8f8f8')
    mode = " [ESCAPE]" if escape_cd > 0 else ""
    tag  = " [GOAL REACHED]" if done_flag else f"  dist={np.linalg.norm(GOAL-pos):.2f}m"
    ax.set_title(f"Step {step}/{N_STEPS}{tag}{mode}  escapes={escapes}", fontsize=9)
    for cx, cy, r in OBS:
        ax.add_patch(patches.Circle((cx,cy), r, color='tomato', alpha=0.75, zorder=2))
    pts = np.array(path)
    ax.plot(pts[:,0], pts[:,1], '-', color='royalblue', alpha=0.4, lw=1, zorder=3)
    ax.add_patch(patches.Circle(GOAL, GOAL_R, color='lime', alpha=0.2, zorder=4))
    ax.plot(*GOAL,  'g*', ms=14, zorder=5, label='Goal')
    ax.plot(*START, 'ks', ms=8,  zorder=5, label='Start')
    ax.plot(*pos,   'bo', ms=11, zorder=6)
    spd = np.linalg.norm(vel)
    if spd > 0.05:
        ax.annotate('', xy=pos+vel/spd*0.45, xytext=pos,
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.legend(loc='upper left', fontsize=8);  ax.grid(alpha=0.2)
    fa = f_att(pos);  fr = f_rep(pos);  ft = fa + fr
    norm_ft = np.linalg.norm(ft)
    if norm_ft > MAX_SPD:
        ft = ft / norm_ft * MAX_SPD
    ax2.set_xlim(-5, 5);  ax2.set_ylim(-5, 5);  ax2.set_aspect('equal')
    ax2.set_title("Force Decomposition at Robot", fontsize=9)
    ax2.axhline(0,color='grey',lw=0.5);  ax2.axvline(0,color='grey',lw=0.5)
    O = np.zeros(2)
    for vec, col, lbl in [(fa,'green','F_att'),(fr,'red','F_rep'),(ft,'blue','F_total')]:
        if np.linalg.norm(vec) > 0.01:
            ax2.annotate('', xy=vec*0.75, xytext=O,
                         arrowprops=dict(arrowstyle='->', color=col, lw=2.5))
            off = vec*0.8 + np.array([0.1, 0.1])
            ax2.text(off[0], off[1], lbl, color=col, fontsize=8)
    ax2.grid(alpha=0.3);  ax2.set_xlabel('Fx');  ax2.set_ylabel('Fy')
    plt.tight_layout()

final_step = N_STEPS
for step in range(N_STEPS):
    noise_v = rng.normal(0, NOISE, 2)

    if escape_cd > 0:
        vel = f_att(pos) + noise_v   # ignore repulsion
        escape_cd -= 1
    else:
        fa = f_att(pos);  fr = f_rep(pos)
        vel = fa + fr + noise_v
        spd = np.linalg.norm(vel)
        if spd > MAX_SPD:
            vel = vel / spd * MAX_SPD

        if len(path) > STUCK_WIN:
            progress = np.linalg.norm(np.array(path[-1]) - np.array(path[-STUCK_WIN]))
            if progress < STUCK_THR:
                escape_cd = ESCAPE_N
                escapes += 1
                print(f"  step {step:3d}: stuck (progress={progress:.2f}m) -> escape {escapes}")

    pos = np.clip(pos + vel * DT, 0.0, ARENA)
    path.append(pos.copy())

    if np.linalg.norm(pos - GOAL) < GOAL_R:
        done = True;  final_step = step
        break

    if step % DRAW_N == 0:
        draw(step, pos, vel, done)
        if INTERACTIVE:
            plt.pause(0.04)

draw(final_step, pos, vel, done)
dist = sum(np.linalg.norm(np.array(path[i+1])-np.array(path[i]))
           for i in range(len(path)-1))
if done:
    print(f"Goal reached at step {final_step}! Path length: {dist:.2f} m  escapes used: {escapes}")
else:
    print(f"Max steps. Dist to goal: {np.linalg.norm(pos-GOAL):.2f} m  escapes used: {escapes}")

plt.savefig("ROBOTICS/outputs/obstacle_avoidance.png", dpi=90, bbox_inches='tight')
print("Saved: ROBOTICS/outputs/obstacle_avoidance.png")
if INTERACTIVE:
    plt.ioff();  plt.show()
plt.close()
