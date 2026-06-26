"""
Formation Control Swarm (Virtual Structure)
============================================
What it does:
  Simulates 5 robots (1 leader + 4 followers) moving in a V-formation.
  The leader follows a figure-8 target path.  Each follower tries to
  maintain a fixed offset from the leader using PD control:
    desired_pos[i] = leader_pos + OFFSET[i]
    error[i]       = desired_pos[i] - current_pos[i]
    velocity[i]    = K_p * error[i] - K_d * current_vel[i]
  A mild inter-robot repulsion prevents collisions when robots get too close
  while recovering from formation after disturbances.

What it teaches:
  - Virtual Structure approach: formation defined by offsets from a reference
  - PD (Proportional-Derivative) formation control
  - Formation error metric: how spread-out the robots are from ideal positions
  - Comparison with other formation approaches (Behaviour-based, Leader-follower)
  - Why formation cohesion and collision avoidance can conflict

Controls: None -- auto-runs for N_STEPS steps.
Animation: Live matplotlib window. Set MPLBACKEND=Agg to skip display (headless).
Output: ROBOTICS/outputs/formation_control.png
RAM: ~60 MB | Time: ~7s with display, <1s headless
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
N_ROBOTS  = 5
N_STEPS   = 400
DT        = 0.05
MAX_SPD   = 2.5
K_P       = 3.5     # PD proportional gain
K_D       = 0.6     # PD derivative (damping)
K_REP     = 1.5     # inter-robot repulsion gain
D_REP     = 1.2     # repulsion activation distance
K_LEAD    = 4.0     # leader P-control gain toward target
DRAW_N    = 3

# V-formation offsets (world-frame, relative to leader position)
# Leader is robot 0; offsets for followers 1-4
OFFSETS = np.array([
    [0.0,    0.0 ],   # leader (reference)
    [-1.8,  -1.8 ],   # follower left-1
    [ 1.8,  -1.8 ],   # follower right-1
    [-3.6,  -3.6 ],   # follower left-2
    [ 3.6,  -3.6 ],   # follower right-2
])

# ─── FIGURE-8 TARGET PATH ─────────────────────────────────────────────────────
def target_pos(t):
    """Leader target: Lissajous figure-8 centred at (5,5)."""
    return np.array([
        5.0 + 4.0 * np.sin(t * 0.4),
        5.0 + 2.5 * np.sin(t * 0.8),
    ])

# ─── INITIAL POSITIONS ────────────────────────────────────────────────────────
# Robots start near (1,1) in rough V-shape with small random offsets
init_center = np.array([1.5, 1.5])
pos = np.array([init_center + OFFSETS[i] + rng.uniform(-0.3, 0.3, 2)
                for i in range(N_ROBOTS)])
vel = np.zeros((N_ROBOTS, 2))
paths = [[p.copy()] for p in pos]
form_errors = []

# ─── SIMULATION ───────────────────────────────────────────────────────────────
print("Formation Control Swarm (V-formation)")
print("=" * 45)

COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
LABELS = ['Leader', 'Follow-L1', 'Follow-R1', 'Follow-L2', 'Follow-R2']

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
fig.suptitle("Formation Control -- V-Formation (Virtual Structure)", fontsize=11)
if INTERACTIVE:
    plt.ion()

def draw(step, t):
    ax.clear();  ax2.clear()
    ax.set_xlim(-1, 11);  ax.set_ylim(-1, 11)
    ax.set_aspect('equal');  ax.set_facecolor('#f5f5f5')
    tpos = target_pos(t)
    ax.set_title(f"Step {step}/{N_STEPS}  "
                 f"form_err={form_errors[-1]:.2f}m  "
                 f"target=({tpos[0]:.1f},{tpos[1]:.1f})", fontsize=9)

    # Target + leader goal
    ax.plot(*tpos, 'r*', ms=16, zorder=5, label='Target')

    # Paths
    for i in range(N_ROBOTS):
        pt = np.array(paths[i])
        ax.plot(pt[:,0], pt[:,1], '-', color=COLORS[i], alpha=0.2, lw=1)

    # Desired formation positions
    for i in range(1, N_ROBOTS):
        dp = pos[0] + OFFSETS[i]
        ax.plot(*dp, '+', ms=10, color=COLORS[i], alpha=0.5, zorder=4)

    # Formation outline (V-shape connecting robots)
    for i in [3, 1, 0, 2, 4]:
        pass   # just draw connections
    v_order = [3, 1, 0, 2, 4]
    vp = np.array([pos[i] for i in v_order])
    ax.plot(vp[:,0], vp[:,1], '-', color='grey', alpha=0.4, lw=2, zorder=3)

    # Robots
    for i in range(N_ROBOTS):
        ax.plot(*pos[i], 'o', ms=14, color=COLORS[i], zorder=6,
                label=LABELS[i], markeredgecolor='black', markeredgewidth=0.5)
        spd = np.linalg.norm(vel[i])
        if spd > 0.05:
            ax.annotate('', xy=pos[i]+vel[i]/spd*0.4, xytext=pos[i],
                        arrowprops=dict(arrowstyle='->', color=COLORS[i], lw=1.5))

    ax.legend(loc='lower right', fontsize=7, ncol=2);  ax.grid(alpha=0.2)

    # Formation error over time
    ax2.plot(form_errors, color='steelblue', lw=1.5)
    ax2.axhline(0, color='green', lw=0.5, linestyle='--')
    ax2.set_title("Formation Error (avg deviation from ideal position)", fontsize=9)
    ax2.set_xlabel("Step");  ax2.set_ylabel("Error (m)")
    ax2.set_ylim(0, max(form_errors)*1.2 + 0.1)
    ax2.grid(alpha=0.3)
    plt.tight_layout()

for step in range(N_STEPS):
    t = step * DT

    # ── Leader: PD toward figure-8 target ────────────────────────────────────
    tgt = target_pos(t)
    err_lead = tgt - pos[0]
    vel[0] = K_LEAD * err_lead - K_D * vel[0]
    spd = np.linalg.norm(vel[0])
    if spd > MAX_SPD:
        vel[0] = vel[0] / spd * MAX_SPD

    # ── Followers: PD toward desired offset position ──────────────────────────
    for i in range(1, N_ROBOTS):
        desired = pos[0] + OFFSETS[i]
        err = desired - pos[i]
        vel[i] = K_P * err - K_D * vel[i]

        # Inter-robot repulsion (avoid collisions)
        for j in range(N_ROBOTS):
            if j == i:
                continue
            diff = pos[i] - pos[j]
            d = np.linalg.norm(diff)
            if 0 < d < D_REP:
                vel[i] += K_REP * (1/d - 1/D_REP) * diff / d

        spd = np.linalg.norm(vel[i])
        if spd > MAX_SPD:
            vel[i] = vel[i] / spd * MAX_SPD

    # ── Integrate positions ───────────────────────────────────────────────────
    pos += vel * DT
    pos = np.clip(pos, 0.0, 10.0)
    for i in range(N_ROBOTS):
        paths[i].append(pos[i].copy())

    # ── Formation error: average distance from ideal offsets ─────────────────
    err_total = np.mean([np.linalg.norm(pos[i] - (pos[0] + OFFSETS[i]))
                         for i in range(1, N_ROBOTS)])
    form_errors.append(err_total)

    if step % DRAW_N == 0:
        draw(step, t)
        if INTERACTIVE:
            plt.pause(0.04)

draw(N_STEPS-1, N_STEPS*DT)
avg_err = np.mean(form_errors[-50:])
print(f"Final average formation error (last 50 steps): {avg_err:.3f} m")
print(f"Formation maintained within {avg_err*100/3.6:.0f}% of inter-robot spacing")

plt.savefig("ROBOTICS/outputs/formation_control.png", dpi=90, bbox_inches='tight')
print("Saved: ROBOTICS/outputs/formation_control.png")
if INTERACTIVE:
    plt.ioff();  plt.show()
plt.close()
