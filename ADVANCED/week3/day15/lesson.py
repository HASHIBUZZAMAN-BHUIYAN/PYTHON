# Advanced Day 15 — Robotics Foundations: 2D Kinematics
# ~50 MB RAM, <5s on CPU

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.random.seed(42)

# ─── 1. COORDINATE FRAMES ─────────────────────────────────────────────────────
print("=== 1. Coordinate Frames ===")
print("""
A coordinate frame defines an origin point and set of axes.
Robots have multiple frames:
  - World frame   (fixed, global reference)
  - Base frame    (robot base)
  - Link frames   (each joint)
  - End-effector  (tool tip)

A 2D point in frame A can be expressed in frame B using a transformation T:
  T = [R  t]   where R = 2x2 rotation matrix, t = 2D translation
      [0  1]
""")

def rotation_matrix_2d(angle_rad):
    """2D rotation matrix."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])

def transform_2d(angle_rad, tx, ty):
    """3×3 homogeneous transformation matrix for 2D."""
    R = rotation_matrix_2d(angle_rad)
    return np.array([
        [R[0,0], R[0,1], tx],
        [R[1,0], R[1,1], ty],
        [0,      0,      1 ]
    ])

# Rotate a point 45° and translate
p = np.array([1., 0., 1.])   # homogeneous
T = transform_2d(np.pi/4, 2., 1.)
p_transformed = T @ p
print(f"Original point: {p[:2]}")
print(f"After 45° rotation + translation (2,1): {p_transformed[:2].round(3)}")

# ─── 2. TWO-LINK ROBOT ARM — FORWARD KINEMATICS ──────────────────────────────
print("\n=== 2. Two-Link Robot Arm — Forward Kinematics ===")
print("""
Forward Kinematics: Given joint angles → compute end-effector position

For a 2-link arm (link lengths L1, L2, joint angles θ1, θ2):
  x = L1*cos(θ1) + L2*cos(θ1+θ2)
  y = L1*sin(θ1) + L2*sin(θ1+θ2)
""")

L1, L2 = 1.0, 0.8   # link lengths

def forward_kinematics_2link(theta1, theta2, L1=L1, L2=L2):
    """Returns joint 1 pos, joint 2 pos, and end-effector pos."""
    origin = np.array([0., 0.])
    j1 = np.array([L1 * np.cos(theta1), L1 * np.sin(theta1)])
    ee = j1 + np.array([L2 * np.cos(theta1 + theta2), L2 * np.sin(theta1 + theta2)])
    return origin, j1, ee

# Test
theta1, theta2 = np.pi/4, np.pi/3
o, j1, ee = forward_kinematics_2link(theta1, theta2)
print(f"θ1={np.degrees(theta1):.0f}°, θ2={np.degrees(theta2):.0f}°")
print(f"Origin: {o}")
print(f"Joint 1: {j1.round(3)}")
print(f"End-effector: {ee.round(3)}")

# ─── 3. WORKSPACE VISUALIZATION ──────────────────────────────────────────────
print("\n=== 3. Workspace Visualization ===")
# Trace the reachable workspace
workspace_x, workspace_y = [], []
for t1 in np.linspace(0, 2*np.pi, 50):
    for t2 in np.linspace(-np.pi, np.pi, 50):
        _, _, end = forward_kinematics_2link(t1, t2)
        workspace_x.append(end[0])
        workspace_y.append(end[1])

# ─── 4. INVERSE KINEMATICS (SIMPLE 2-LINK) ───────────────────────────────────
print("\n=== 4. Inverse Kinematics ===")
print("""
Inverse Kinematics: Given target end-effector pos → find joint angles.

For 2-link arm (geometric solution):
  cos(θ2) = (x²+y²-L1²-L2²) / (2·L1·L2)
  θ2 = atan2(±√(1-cos²θ2), cos(θ2))   [elbow up/down solution]
  θ1 = atan2(y,x) - atan2(L2·sin(θ2), L1+L2·cos(θ2))
""")

def inverse_kinematics_2link(x, y, L1=L1, L2=L2, elbow_up=True):
    """Computes joint angles for given end-effector position."""
    r2 = x**2 + y**2
    cos_t2 = (r2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_t2 = np.clip(cos_t2, -1, 1)   # numerical stability
    sign = 1 if elbow_up else -1
    sin_t2 = sign * np.sqrt(1 - cos_t2**2)
    theta2 = np.arctan2(sin_t2, cos_t2)
    theta1 = np.arctan2(y, x) - np.arctan2(L2*sin_t2, L1+L2*cos_t2)
    return theta1, theta2

target = np.array([1.2, 0.5])
t1, t2 = inverse_kinematics_2link(*target)
_, _, ee_check = forward_kinematics_2link(t1, t2)
print(f"Target: {target}")
print(f"IK solution: θ1={np.degrees(t1):.2f}°, θ2={np.degrees(t2):.2f}°")
print(f"FK check: {ee_check.round(4)}  ← should match target")

# ─── 5. VISUALIZATION ─────────────────────────────────────────────────────────
def draw_arm(ax, theta1, theta2, color="steelblue", label=""):
    o, j1, ee = forward_kinematics_2link(theta1, theta2)
    ax.plot([o[0],j1[0]], [o[1],j1[1]], "o-", color=color, linewidth=3, markersize=8, label=label)
    ax.plot([j1[0],ee[0]], [j1[1],ee[1]], "o-", color=color, linewidth=3, markersize=8)
    ax.plot(*ee, "*", color="gold", markersize=15, zorder=5)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Workspace
axes[0].scatter(workspace_x, workspace_y, s=1, alpha=0.1, color="steelblue")
axes[0].set_aspect("equal"); axes[0].set_title("Reachable Workspace")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y"); axes[0].grid(True, alpha=0.3)

# Arm configurations
axes[1].set_xlim(-2.2, 2.2); axes[1].set_ylim(-2.2, 2.2); axes[1].set_aspect("equal")
axes[1].grid(True, alpha=0.3); axes[1].set_title("IK Solutions (elbow up/down)")
t1_up, t2_up = inverse_kinematics_2link(*target, elbow_up=True)
t1_dn, t2_dn = inverse_kinematics_2link(*target, elbow_up=False)
draw_arm(axes[1], t1_up, t2_up, "steelblue", "Elbow up")
draw_arm(axes[1], t1_dn, t2_dn, "tomato", "Elbow down")
axes[1].plot(*target, "k+", markersize=15, markeredgewidth=2, label="Target")
axes[1].legend()

plt.tight_layout(); plt.savefig("kinematics.png", dpi=80)
print("\nSaved kinematics.png")

# ─── 6. SIMPLE TRAJECTORY FOLLOWING ──────────────────────────────────────────
print("\n=== 5. Trajectory Following Demo ===")
# Move end-effector along a circular path
angles_path = np.linspace(0, 2*np.pi, 40)
r_path = 0.6
path_x = 0.8 + r_path * np.cos(angles_path)
path_y = 0.0 + r_path * np.sin(angles_path)

traj_thetas = []
for px, py in zip(path_x, path_y):
    try:
        t1, t2 = inverse_kinematics_2link(px, py)
        traj_thetas.append((t1, t2))
    except:
        pass
print(f"Generated {len(traj_thetas)} trajectory waypoints.")
print("(Run animation in mini_project.py for animated visualization)")
