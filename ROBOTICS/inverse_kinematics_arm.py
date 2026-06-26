"""
2-Link Robot Arm Inverse Kinematics
======================================
What it does:
  Simulates a 2-link planar robot arm moving through a sequence of 6 target
  positions using ANALYTICAL inverse kinematics.  At each target the arm
  smoothly interpolates its joint angles (shoulder theta1 and elbow theta2)
  from the current configuration to the IK solution.

  IK solution (elbow-up convention):
    cos(theta2) = (x^2 + y^2 - L1^2 - L2^2) / (2*L1*L2)
    theta2 = arccos(clamp(cos_t2, -1, 1))
    theta1 = atan2(y, x) - atan2(L2*sin(theta2), L1 + L2*cos(theta2))

  The animation shows:
    LEFT:  arm in the workspace, target position, reachable boundary circle
    RIGHT: joint angle trajectories over time (theta1 in blue, theta2 in green)

What it teaches:
  - Forward kinematics: joint angles -> end-effector (x, y)
  - Inverse kinematics (2-link analytical): (x, y) -> joint angles
  - Elbow-up vs. elbow-down solutions (two valid IK solutions, one chosen)
  - Workspace limits: why some targets are unreachable (outside annulus)
  - Smooth joint interpolation: cosine ease-in/ease-out for natural motion

Controls: None -- auto-runs through all 6 targets and exits.
Animation: Live matplotlib window. Set MPLBACKEND=Agg to skip display (headless).
Output: ROBOTICS/outputs/ik_arm.png
RAM: ~50 MB | Time: ~6s with display, <1s headless
"""

import os
import numpy as np
import matplotlib
INTERACTIVE = matplotlib.get_backend().lower() != 'agg'
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs("ROBOTICS/outputs", exist_ok=True)

# ─── ARM CONSTANTS ────────────────────────────────────────────────────────────
L1 = 3.5      # upper-arm link length (m)
L2 = 2.5      # forearm link length  (m)
BASE = np.array([0.0, 0.0])
FRAMES_PER_MOVE = 60   # frames to interpolate between configs
DRAW_N = 2

# Six target positions in the reachable workspace (annulus L1-L2 to L1+L2)
TARGETS = np.array([
    [5.0,  1.0],
    [3.0,  4.5],
    [-2.0, 5.0],
    [-5.5, 0.5],
    [-1.0,-4.5],
    [4.5, -2.5],
])

# ─── KINEMATICS ───────────────────────────────────────────────────────────────
def fk(t1, t2):
    """Forward kinematics: joint angles -> (elbow_pos, end_pos)."""
    elbow = BASE + L1 * np.array([np.cos(t1), np.sin(t1)])
    end   = elbow + L2 * np.array([np.cos(t1+t2), np.sin(t1+t2)])
    return elbow, end

def ik(target):
    """Analytical IK; returns (theta1, theta2) or None if unreachable."""
    x, y  = target - BASE
    d     = np.hypot(x, y)
    # Clamp to reachable annulus
    d_min = abs(L1 - L2) + 1e-4
    d_max = L1 + L2 - 1e-4
    if d > d_max:
        x, y = x/d*d_max, y/d*d_max
        d = d_max
    elif d < d_min:
        x, y = x/d*d_min, y/d*d_min
        d = d_min
    cos_t2 = (x**2 + y**2 - L1**2 - L2**2) / (2*L1*L2)
    cos_t2 = np.clip(cos_t2, -1.0, 1.0)
    t2 = np.arccos(cos_t2)
    t1 = np.arctan2(y, x) - np.arctan2(L2*np.sin(t2), L1 + L2*np.cos(t2))
    return t1, t2

def lerp_angle(a, b, t):
    """Shortest-arc interpolation between two angles."""
    diff = ((b - a) + np.pi) % (2*np.pi) - np.pi
    return a + diff * t

def ease(t):
    """Cosine ease-in/ease-out: t in [0,1] -> smooth [0,1]."""
    return 0.5 * (1 - np.cos(np.pi * t))

# ─── SIMULATION ───────────────────────────────────────────────────────────────
print("2-Link Arm Inverse Kinematics")
print("=" * 45)

t1, t2  = np.pi/6, np.pi/4    # initial configuration
history_t1, history_t2 = [t1], [t2]
all_ends = [fk(t1, t2)[1].copy()]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("2-Link Robot Arm - Analytical Inverse Kinematics", fontsize=11)
if INTERACTIVE:
    plt.ion()

EXTENT = L1 + L2 + 0.5

def draw(tgt_idx, frame_global, t1, t2, target):
    ax.clear();  ax2.clear()

    ax.set_xlim(-EXTENT, EXTENT);  ax.set_ylim(-EXTENT, EXTENT)
    ax.set_aspect('equal');  ax.set_facecolor('#f0f0f0')
    ax.set_title(f"Workspace  target {tgt_idx+1}/{len(TARGETS)}", fontsize=9)

    theta_ring = np.linspace(0, 2*np.pi, 200)
    ax.plot((L1+L2)*np.cos(theta_ring), (L1+L2)*np.sin(theta_ring),
            'g--', alpha=0.3, lw=1, label='Outer reach')
    inner = abs(L1-L2)
    ax.plot(inner*np.cos(theta_ring), inner*np.sin(theta_ring),
            'r--', alpha=0.3, lw=1, label='Inner reach')

    if len(all_ends) > 1:
        ae = np.array(all_ends)
        ax.plot(ae[:,0], ae[:,1], '-', color='slategrey', alpha=0.3, lw=1)

    elbow, end = fk(t1, t2)
    ax.plot([BASE[0], elbow[0]], [BASE[1], elbow[1]],
            '-o', color='royalblue', lw=5, ms=8, zorder=4)
    ax.plot([elbow[0], end[0]], [elbow[1], end[1]],
            '-o', color='steelblue', lw=4, ms=8, zorder=4)
    ax.plot(*end,  'ro', ms=10, zorder=5, label='End-effector')
    ax.plot(*BASE, 'ks', ms=10, zorder=5, label='Base')

    ax.plot(*target, 'g*', ms=14, zorder=6, label='Target')
    ax.add_patch(patches.Circle(target, 0.15, color='green', alpha=0.2))

    ax.set_xlabel(f"theta1={np.degrees(t1):.1f} deg   theta2={np.degrees(t2):.1f} deg   "
                  f"|ee-target|={np.linalg.norm(end-target):.3f}m", fontsize=8)
    ax.legend(loc='upper left', fontsize=7);  ax.grid(alpha=0.2)

    ax2.set_title("Joint Angles Over Time", fontsize=9)
    ax2.plot(np.degrees(history_t1), color='royalblue', label='theta1 (shoulder)', lw=1.5)
    ax2.plot(np.degrees(history_t2), color='forestgreen', label='theta2 (elbow)', lw=1.5)
    ax2.axvline(frame_global, color='red', lw=1, alpha=0.5)
    ax2.set_xlabel("Frame");  ax2.set_ylabel("Angle (deg)")
    ax2.legend(fontsize=8);  ax2.grid(alpha=0.3)
    plt.tight_layout()

frame_global = 0
for ti, target in enumerate(TARGETS):
    t1_tgt, t2_tgt = ik(target)
    t1_src, t2_src = t1, t2
    print(f"  Target {ti+1}: {target}  IK -> t1={np.degrees(t1_tgt):.1f}d  t2={np.degrees(t2_tgt):.1f}d")
    for f in range(FRAMES_PER_MOVE):
        alpha = ease(f / FRAMES_PER_MOVE)
        t1 = lerp_angle(t1_src, t1_tgt, alpha)
        t2 = lerp_angle(t2_src, t2_tgt, alpha)
        history_t1.append(t1);  history_t2.append(t2)
        all_ends.append(fk(t1, t2)[1].copy())
        frame_global += 1
        if f % DRAW_N == 0:
            draw(ti, frame_global, t1, t2, target)
            if INTERACTIVE:
                plt.pause(0.03)

draw(len(TARGETS)-1, frame_global, t1, t2, TARGETS[-1])
print("Cycle complete. Final EE:", np.round(fk(t1, t2)[1], 3))

plt.savefig("ROBOTICS/outputs/ik_arm.png", dpi=90, bbox_inches='tight')
print("Saved: ROBOTICS/outputs/ik_arm.png")
if INTERACTIVE:
    plt.ioff();  plt.show()
plt.close()
