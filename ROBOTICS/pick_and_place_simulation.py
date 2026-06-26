"""
Pick-and-Place Robot Arm Simulation
=====================================
What it does:
  Extends the 2-link IK arm with a pick-and-place task:
    - 4 objects are scattered in the workspace
    - 1 drop zone is fixed on the right
    - A finite state machine (FSM) drives the arm through a cycle for each
      object: MOVE_TO_OBJECT -> GRASP -> LIFT -> MOVE_TO_DROP -> PLACE
    - While in the GRASP and LIFT states the object visually sticks to the
      end-effector position (simulating a suction-cup or gripper)
    - After all objects are placed, the arm returns to its home configuration

  Arm dynamics: same cosine-eased joint-angle interpolation as the IK script.
  Gripper: binary open/closed -- no finger simulation.

What it teaches:
  - FSM (Finite State Machine) as a task execution layer above motion control
  - How to chain multiple robot capabilities: IK + grasping + transport
  - Why real robots need separate motion planning, task planning, and perception
  - The concept of a "home position" and safe intermediate waypoints
  - How object attachments are modelled in simulation (kinematic coupling)

Controls: None -- auto-runs through all objects.
Animation: Live matplotlib window. Set MPLBACKEND=Agg to skip display (headless).
Output: ROBOTICS/outputs/pick_and_place.png
RAM: ~50 MB | Time: ~8s with display, <1s headless
"""

import os
import numpy as np
import matplotlib
INTERACTIVE = matplotlib.get_backend().lower() != 'agg'
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs("ROBOTICS/outputs", exist_ok=True)

# ─── ARM ──────────────────────────────────────────────────────────────────────
L1, L2 = 3.5, 2.5
BASE   = np.array([0.0, 0.0])

def fk(t1, t2):
    elbow = BASE + L1 * np.array([np.cos(t1), np.sin(t1)])
    end   = elbow + L2 * np.array([np.cos(t1+t2), np.sin(t1+t2)])
    return elbow, end

def ik(target):
    x, y  = target - BASE
    d     = np.hypot(x, y)
    d_max = L1 + L2 - 1e-4
    d_min = abs(L1 - L2) + 1e-4
    if d > d_max: x, y = x/d*d_max, y/d*d_max
    elif d < d_min: x, y = x/d*d_min, y/d*d_min
    cos_t2 = np.clip((x**2+y**2 - L1**2 - L2**2)/(2*L1*L2), -1.0, 1.0)
    t2 = np.arccos(cos_t2)
    t1 = np.arctan2(y, x) - np.arctan2(L2*np.sin(t2), L1+L2*np.cos(t2))
    return t1, t2

def lerp_angle(a, b, t):
    diff = ((b-a)+np.pi) % (2*np.pi) - np.pi
    return a + diff*t

def ease(t):
    return 0.5*(1 - np.cos(np.pi*t))

# ─── SCENE ────────────────────────────────────────────────────────────────────
OBJECTS = [
    {'pos': np.array([ 4.5,  2.0]), 'color': '#e74c3c', 'label': 'Red box'},
    {'pos': np.array([-1.5,  5.0]), 'color': '#3498db', 'label': 'Blue box'},
    {'pos': np.array([-4.0, -1.0]), 'color': '#f39c12', 'label': 'Orange box'},
    {'pos': np.array([ 2.0, -4.5]), 'color': '#2ecc71', 'label': 'Green box'},
]
DROP_ZONE = np.array([5.5, -1.0])
HOME_T1, HOME_T2 = np.pi/2, np.pi/3   # arm resting position (pointing up)
LIFT_HEIGHT = np.array([0.0, 1.2])    # how far above object to lift

FRAMES_MOVE  = 45
FRAMES_GRASP = 20
DRAW_N = 2

# ─── FSM STATES ───────────────────────────────────────────────────────────────
STATES = ['IDLE', 'MOVE_TO_OBJ', 'GRASP', 'LIFT', 'MOVE_TO_DROP', 'PLACE', 'DONE']

# ─── SIMULATION ───────────────────────────────────────────────────────────────
print("Pick-and-Place Simulation")
print("=" * 45)

t1, t2 = HOME_T1, HOME_T2
held_obj   = None    # index of currently held object
placed     = []      # indices of placed objects
placed_pos = []      # positions at drop zone
state      = 'IDLE'

fig, ax = plt.subplots(figsize=(9, 9))
if INTERACTIVE:
    plt.ion()

EXTENT = L1 + L2 + 1.0

def draw(state_str, t1, t2, gripper_closed):
    ax.clear()
    ax.set_xlim(-EXTENT, EXTENT);  ax.set_ylim(-EXTENT, EXTENT)
    ax.set_aspect('equal');  ax.set_facecolor('#f0f0f0')
    ax.set_title(f"Pick-and-Place  state: {state_str}  "
                 f"placed {len(placed)}/{len(OBJECTS)}", fontsize=10)

    # Drop zone
    ax.add_patch(patches.Rectangle(DROP_ZONE-0.5, 1.0, 1.0,
                                   facecolor='khaki', edgecolor='goldenrod', lw=2, zorder=2))
    ax.text(*DROP_ZONE, 'DROP\nZONE', ha='center', va='center', fontsize=7, zorder=3)

    # Objects still in scene
    elbow, end = fk(t1, t2)
    for i, obj in enumerate(OBJECTS):
        if i in placed:
            continue
        pos = end if i == held_obj else obj['pos']
        ax.add_patch(patches.FancyBboxPatch(pos-0.25, 0.5, 0.5,
                     boxstyle='round,pad=0.05',
                     facecolor=obj['color'], edgecolor='black', lw=1.5, zorder=4))
        ax.text(pos[0], pos[1], str(i+1), ha='center', va='center',
                fontsize=8, color='white', fontweight='bold', zorder=5)

    # Placed objects at drop zone
    for k, pp in enumerate(placed_pos):
        pobj = OBJECTS[placed[k]]
        ax.add_patch(patches.FancyBboxPatch(pp-0.25, 0.5, 0.5,
                     boxstyle='round,pad=0.05',
                     facecolor=pobj['color'], edgecolor='black', lw=1.5, zorder=4,
                     alpha=0.5))

    # Arm
    ax.plot([BASE[0], elbow[0]], [BASE[1], elbow[1]],
            '-o', color='royalblue', lw=5, ms=8, zorder=6)
    ax.plot([elbow[0], end[0]], [elbow[1], end[1]],
            '-o', color='steelblue', lw=4, ms=8, zorder=6)
    ax.plot(*BASE, 'ks', ms=10, zorder=7)

    # Gripper visual (two small lines)
    gc = 0.15 if gripper_closed else 0.35
    perp = np.array([-np.sin(t1+t2), np.cos(t1+t2)])
    g_color = 'red' if gripper_closed else 'green'
    for sign in [+1, -1]:
        gp = end + sign * gc * perp
        ax.plot([end[0], gp[0]], [end[1], gp[1]], '-', color=g_color, lw=3, zorder=7)
    ax.text(-EXTENT+0.2, -EXTENT+0.2, f"Gripper: {'CLOSED' if gripper_closed else 'OPEN'}",
            fontsize=8, color=g_color)
    ax.grid(alpha=0.2);  plt.tight_layout()

def move_to(t1_src, t2_src, target, n_frames, gripper_closed, label):
    """Interpolate arm to target; return final (t1, t2)."""
    t1_tgt, t2_tgt = ik(target)
    for f in range(n_frames):
        alpha = ease(f / n_frames)
        t1c = lerp_angle(t1_src, t1_tgt, alpha)
        t2c = lerp_angle(t2_src, t2_tgt, alpha)
        if f % DRAW_N == 0:
            draw(label, t1c, t2c, gripper_closed)
            if INTERACTIVE:
                plt.pause(0.03)
    return t1_tgt, t2_tgt

def hold_frames(t1, t2, n_frames, gripper_closed, label):
    """Hold position for n_frames."""
    for f in range(n_frames):
        if f % DRAW_N == 0:
            draw(label, t1, t2, gripper_closed)
            if INTERACTIVE:
                plt.pause(0.03)

for obj_idx, obj in enumerate(OBJECTS):
    print(f"  Object {obj_idx+1}: {obj['label']}  at {obj['pos']}")
    obj_pos  = obj['pos']
    lift_pos = obj_pos + LIFT_HEIGHT

    # MOVE_TO_OBJ
    t1, t2 = move_to(t1, t2, obj_pos, FRAMES_MOVE, False, 'MOVE_TO_OBJ')
    # GRASP
    hold_frames(t1, t2, FRAMES_GRASP, True, 'GRASP')
    held_obj = obj_idx
    print(f"    Grasped object {obj_idx+1}")
    # LIFT
    t1, t2 = move_to(t1, t2, lift_pos, FRAMES_MOVE//2, True, 'LIFT')
    # MOVE_TO_DROP
    t1, t2 = move_to(t1, t2, DROP_ZONE + LIFT_HEIGHT, FRAMES_MOVE, True, 'MOVE_TO_DROP')
    # Lower to drop zone
    t1, t2 = move_to(t1, t2, DROP_ZONE, FRAMES_MOVE//2, True, 'LOWER')
    # PLACE
    hold_frames(t1, t2, FRAMES_GRASP, False, 'PLACE')
    drop_offset = np.array([0.0, (len(placed)) * 0.0])  # stack in place
    placed.append(obj_idx)
    placed_pos.append(DROP_ZONE.copy())
    held_obj = None
    print(f"    Placed at drop zone  ({len(placed)}/{len(OBJECTS)})")
    # Return to home
    t1, t2 = move_to(t1, t2, np.array([0.0, L1+L2-0.5]), FRAMES_MOVE, False, 'RETURN_HOME')

# Final home
t1, t2 = move_to(t1, t2, np.array([0.0, 2.5]), FRAMES_MOVE//2, False, 'DONE')
draw('DONE', t1, t2, False)
print(f"All {len(OBJECTS)} objects placed at drop zone!")

plt.savefig("ROBOTICS/outputs/pick_and_place.png", dpi=90, bbox_inches='tight')
print("Saved: ROBOTICS/outputs/pick_and_place.png")
if INTERACTIVE:
    plt.ioff();  plt.show()
plt.close()
