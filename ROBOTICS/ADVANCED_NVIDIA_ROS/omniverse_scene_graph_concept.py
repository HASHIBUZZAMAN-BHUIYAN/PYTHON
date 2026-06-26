"""
NVIDIA Omniverse Scene Graph / USD Digital Twin (Pure Python + Matplotlib 3D)
=============================================================================
REAL TOOL THIS MODELS:
  NVIDIA Omniverse is a platform for building real-time 3D collaborative
  applications and digital twins.  Its foundation is USD (Universal Scene
  Description), developed by Pixar and now used across the robotics/sim
  industry.  In Omniverse:
    - A "Stage" is the top-level scene container (equivalent to a USD file)
    - A "Prim" (primitive) is any object in the scene: meshes, lights, cameras,
      or just coordinate frames (Xform prims)
    - Each prim has a local transform (position, rotation, scale)
    - The scene is a TREE: child prims inherit their parent's world transform
      (world_T = parent_world_T @ local_T, using homogeneous 4x4 matrices)
    - Omniverse supports live sync: multiple apps (Isaac Sim, Blender, CAD)
      can edit the same scene simultaneously via Nucleus server

  Isaac Sim (part of NVIDIA Omniverse) uses USD as its scene representation
  for robot simulations -- importing robot URDFs, running PhysX physics,
  and streaming sensor data to the Isaac ROS stack.

WHAT THIS SCRIPT DOES (the honest, working version):
  Implements a fully-working scene graph in pure Python with matplotlib 3D:
    - SceneGraph class: a tree of Prims, each with local SE(3) transform
    - Rigid body transform: 4x4 homogeneous matrix (SO(3) rotation + translation)
    - Parent-to-world propagation: world_T[child] = world_T[parent] @ local_T[child]
    - A 5-link robotic arm (base -> shoulder -> upper_arm -> forearm -> wrist -> ee)
      is defined as a hierarchy of Xform prims
    - Joint angles are animated sinusoidally, the graph recomputes world
      transforms each frame, and matplotlib 3D renders the arm
    - A second "camera" prim is attached to the end-effector -- its world
      position is extracted from the scene graph automatically (showing how
      Omniverse tracks sensor positions from the robot kinematic chain)

WHAT REAL OMNIVERSE / USD ADDS (beyond this script):
  - Full photorealistic real-time rendering (RTX, ray tracing on NVIDIA GPU)
  - PhysX physics simulation (rigid bodies, joints, collisions, cloth, fluid)
  - USD file format: .usd/.usda files with prims, attributes, schemas, layers
  - Live collaborative editing: Nucleus server + Omniverse USD Composer
  - Python USD API (pxr.Usd, pxr.UsdGeom) for programmatic scene construction
  - Integration with Isaac Sim for robot training and digital twins
  - Multi-app sync: Blender, Maya, CAD tools all editing the same USD stage
  To learn more: developer.nvidia.com/omniverse | openusd.org

THIS SCRIPT RUNS ON: CPU-only, ~80 MB RAM, Python 3.9+ with numpy + matplotlib.
No NVIDIA GPU, no Omniverse installation, no USD library needed.

Output: ROBOTICS/ADVANCED_NVIDIA_ROS/outputs/omniverse_scenegraph.png
"""

import os, math
import numpy as np
import matplotlib
INTERACTIVE = matplotlib.get_backend().lower() != 'agg'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # registers 3D projection

os.makedirs("ROBOTICS/ADVANCED_NVIDIA_ROS/outputs", exist_ok=True)

# ─── 4x4 HOMOGENEOUS TRANSFORM HELPERS ───────────────────────────────────────
def T_identity():
    return np.eye(4)

def T_translate(x, y, z):
    M = np.eye(4)
    M[:3, 3] = [x, y, z]
    return M

def T_rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]], dtype=float)

def T_rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], dtype=float)

def T_rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)

def origin_in(T):
    """Extract the world-space origin of a transform (last column)."""
    return T[:3, 3]

def apply_T(T, p):
    """Apply 4x4 transform to a 3D point (returned as 3-vec)."""
    return (T @ np.array([p[0], p[1], p[2], 1.0]))[:3]

# ─── SCENE GRAPH ──────────────────────────────────────────────────────────────
class Prim:
    """A node in the USD-style scene graph."""
    def __init__(self, name, parent=None):
        self.name     = name
        self.parent   = parent
        self.children = []
        self.local_T  = T_identity()   # local transform (relative to parent)
        if parent:
            parent.children.append(self)

    def world_T(self):
        """Compute world transform by walking up the tree."""
        if self.parent is None:
            return self.local_T.copy()
        return self.parent.world_T() @ self.local_T

    def world_pos(self):
        return origin_in(self.world_T())

class SceneGraph:
    """Minimal USD Stage equivalent."""
    def __init__(self):
        self.root = Prim("World")
        self.prims = {"World": self.root}

    def define_prim(self, path, parent_path="World"):
        parent = self.prims[parent_path]
        prim   = Prim(path, parent)
        self.prims[path] = prim
        return prim

# ─── BUILD 5-DOF ROBOT ARM SCENE ─────────────────────────────────────────────
# Link lengths (metres)
L_BASE = 0.5;  L1 = 1.0;  L2 = 0.9;  L3 = 0.7;  L4 = 0.4

stage = SceneGraph()

base = stage.define_prim("base", "World")
base.local_T = T_translate(0, 0, 0)         # base at origin

shoulder = stage.define_prim("shoulder", "base")
shoulder.local_T = T_translate(0, 0, L_BASE) @ T_rot_z(0)   # q1

upper_arm = stage.define_prim("upper_arm", "shoulder")
upper_arm.local_T = T_translate(0, 0, L1)   # after q2 rotation (applied at shoulder)

forearm = stage.define_prim("forearm", "upper_arm")
forearm.local_T = T_translate(0, 0, L2)     # after q3 at upper_arm

wrist = stage.define_prim("wrist", "forearm")
wrist.local_T = T_translate(0, 0, L3)

ee = stage.define_prim("end_effector", "wrist")
ee.local_T = T_translate(0, 0, L4)

camera = stage.define_prim("camera", "end_effector")
camera.local_T = T_translate(0.05, 0, 0.1)  # slightly offset from EE tip

# ─── ANIMATION ────────────────────────────────────────────────────────────────
N_FRAMES = 120
DRAW_N   = 4

print("Omniverse Scene Graph / Digital Twin Concept (5-DOF Arm)")
print("=" * 55)

fig = plt.figure(figsize=(14, 6))
ax3d = fig.add_subplot(121, projection='3d')
ax2d = fig.add_subplot(122)
fig.suptitle("Omniverse USD Scene Graph: 5-DoF Robot Arm Digital Twin", fontsize=10)

REACH = L_BASE + L1 + L2 + L3 + L4
ee_positions = []

if INTERACTIVE:
    plt.ion()

def update_joints(frame):
    """Set joint angles from sinusoidal motion profile."""
    t  = frame / N_FRAMES * 2 * math.pi
    q1 = 0.8 * math.sin(t)
    q2 = 0.6 * math.sin(t * 1.3 + 0.5)
    q3 = 0.5 * math.sin(t * 0.7 - 0.3)
    q4 = 0.4 * math.cos(t * 1.1)

    # Apply joint angles by recomposing local_T for each link
    shoulder.local_T  = T_translate(0, 0, L_BASE) @ T_rot_z(q1) @ T_rot_y(q2)
    upper_arm.local_T = T_translate(0, 0, L1)     @ T_rot_y(q3)
    forearm.local_T   = T_translate(0, 0, L2)     @ T_rot_y(q4)
    wrist.local_T     = T_translate(0, 0, L3)
    ee.local_T        = T_translate(0, 0, L4)
    camera.local_T    = T_translate(0.05, 0, 0.1)

def draw(frame):
    ax3d.clear();  ax2d.clear()
    update_joints(frame)

    # Collect world positions of each link
    prims_ordered = [base, shoulder, upper_arm, forearm, wrist, ee]
    pts = np.array([p.world_pos() for p in prims_ordered])
    cam_pos = camera.world_pos()
    ee_pos  = ee.world_pos()

    # 3D arm
    ax3d.plot(pts[:,0], pts[:,1], pts[:,2], 'o-',
              color='royalblue', ms=8, lw=4, zorder=5, label='Arm links')
    ax3d.scatter(*cam_pos, s=80, c='orange', zorder=6, label='Camera (EE child)')
    ax3d.scatter(*pts[0],  s=100, c='black', zorder=7)  # base

    ax3d.set_xlim(-REACH, REACH);  ax3d.set_ylim(-REACH, REACH)
    ax3d.set_zlim(0, REACH*1.2)
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
    ax3d.set_title(f"3D Scene Graph  frame {frame}/{N_FRAMES}", fontsize=9)
    ax3d.legend(fontsize=8, loc='upper right')

    # EE trajectory (top-down XY)
    ax2d.set_xlim(-REACH, REACH);  ax2d.set_ylim(-REACH, REACH)
    ax2d.set_aspect('equal');  ax2d.set_facecolor('#f8f8f8')
    ax2d.set_title("End-Effector XY Trace (extracted from scene graph)", fontsize=9)
    if len(ee_positions) > 1:
        ep = np.array(ee_positions)
        ax2d.plot(ep[:,0], ep[:,1], '-', color='steelblue', alpha=0.5, lw=1.5)
    ax2d.plot(*ee_pos[:2], 'bo', ms=8, zorder=5, label='EE')
    ax2d.plot(*cam_pos[:2], 's', color='orange', ms=7, zorder=5, label='Camera')

    # Scene graph tree (text)
    tree_lines = [
        "USD Scene Graph (live):",
        f"  /World",
        f"    /base         pos={tuple(np.round(base.world_pos(),2))}",
        f"    /shoulder     pos={tuple(np.round(shoulder.world_pos(),2))}",
        f"    /upper_arm    pos={tuple(np.round(upper_arm.world_pos(),2))}",
        f"    /forearm      pos={tuple(np.round(forearm.world_pos(),2))}",
        f"    /wrist        pos={tuple(np.round(wrist.world_pos(),2))}",
        f"    /end_effector pos={tuple(np.round(ee.world_pos(),2))}",
        f"    /camera       pos={tuple(np.round(cam_pos,2))}",
    ]
    ax2d.text(0.02, 0.98, "\n".join(tree_lines), transform=ax2d.transAxes,
              fontsize=7, fontfamily='monospace', va='top',
              bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'))
    ax2d.legend(fontsize=8);  ax2d.grid(alpha=0.2)
    plt.tight_layout()

for frame in range(N_FRAMES):
    update_joints(frame)
    ee_positions.append(ee.world_pos().copy())
    if frame % DRAW_N == 0:
        draw(frame)
        if INTERACTIVE:
            plt.pause(0.04)

draw(N_FRAMES - 1)
final_ee = ee.world_pos()
final_cam = camera.world_pos()
print(f"Animation complete: {N_FRAMES} frames, {len(stage.prims)} prims in scene")
print(f"Final EE world pos:  {np.round(final_ee, 3)}")
print(f"Final cam world pos: {np.round(final_cam, 3)}")
print("  (Camera pos derived automatically from scene graph -- no manual math)")

plt.savefig("ROBOTICS/ADVANCED_NVIDIA_ROS/outputs/omniverse_scenegraph.png",
            dpi=90, bbox_inches='tight')
print("Saved: ROBOTICS/ADVANCED_NVIDIA_ROS/outputs/omniverse_scenegraph.png")
if INTERACTIVE:
    plt.ioff();  plt.show()
plt.close()
