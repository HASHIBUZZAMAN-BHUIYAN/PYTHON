"""
NVIDIA Isaac GR00T-Style Robot Policy (Imitation Learning, Pure NumPy)
=======================================================================
REAL TOOL THIS MODELS:
  NVIDIA Isaac GR00T is a robot foundation model -- a large neural network
  pre-trained on vast amounts of real and simulated robot demonstration data.
  It implements the "observation -> policy -> action" loop that is common to
  all modern learning-based robot control systems:
    - Observation: what the robot currently perceives (camera frames, joint
      angles, task instructions, etc.)
    - Policy network: a large transformer or diffusion model that maps
      observations to action distributions
    - Action: the robot's next motor command (joint velocities, gripper state)
  GR00T is designed to run on NVIDIA Jetson or full NVIDIA GPU setups and
  interfaces with NVIDIA Isaac Sim (the physics simulation) or real hardware
  via the Isaac ROS stack.

WHAT THIS SCRIPT DOES (the honest, working version):
  Implements the CORE CONCEPT -- "observation -> policy -> action" imitation
  learning -- in pure NumPy on a simple 2D navigation task:
    1. DEMONSTRATION COLLECTION: a hand-crafted PD controller (the "expert")
       navigates a 2D robot from random starts to a fixed goal.  Each step
       records an (observation, action) pair.  Observation: [x, y, dx_goal,
       dy_goal, dist_goal].  Action: [vx, vy] velocity command.
    2. POLICY TRAINING: a tiny 3-layer MLP (32 hidden units, ReLU, MSE loss)
       is trained on these demonstrations using mini-batch gradient descent
       implemented from scratch in NumPy.  This is behavioural cloning (BC).
    3. POLICY ROLLOUT: the trained policy drives the robot in a new episode
       (no expert, no PD controller).  Plots show expert trajectories vs
       policy trajectories and the training loss curve.

WHAT REAL GR00T / LEARNING-BASED ROBOT POLICIES ADD:
  - Scale: GR00T is trained on millions of demonstrations and billions of
    image tokens; this MLP has 3 layers and sees 200 demo timesteps
  - Modalities: GR00T takes RGB images, language instructions, and joint
    states as input; this MLP takes 5 scalars
  - Temporal modelling: diffusion-policy or transformer-based action chunking
    for smooth, temporally coherent behaviour; this MLP acts independently
    each step (no memory)
  - Real hardware integration: GR00T interfaces with NVIDIA Isaac ROS,
    real URDF models, physical robot joint controllers
  - To learn more: developer.nvidia.com/isaac  |  gr00t.nvidia.com
    (GR00T paper: "A Foundation Model for Generalizable Robotics")

THIS SCRIPT RUNS ON: CPU-only, ~80 MB RAM, Python 3.9+ with numpy + matplotlib.
No GPU, no NVIDIA hardware, no Isaac installation needed.

Output: ROBOTICS/ADVANCED_NVIDIA_ROS/outputs/groot_policy.png
"""

import os
import numpy as np
import matplotlib
INTERACTIVE = matplotlib.get_backend().lower() != 'agg'
import matplotlib.pyplot as plt

os.makedirs("ROBOTICS/ADVANCED_NVIDIA_ROS/outputs", exist_ok=True)
rng = np.random.default_rng(42)

# ─── ENVIRONMENT ──────────────────────────────────────────────────────────────
ARENA  = 10.0
GOAL   = np.array([8.0, 8.0])
GOAL_R = 0.5
DT     = 0.08
MAX_SPD = 2.0
OBSTACLES = [(3.0, 3.0, 0.6), (6.5, 4.5, 0.7)]   # (cx, cy, r)

def obs_from_state(pos):
    """Construct observation vector from robot position."""
    diff = GOAL - pos
    return np.array([pos[0]/ARENA, pos[1]/ARENA,
                     diff[0]/ARENA, diff[1]/ARENA,
                     np.linalg.norm(diff)/ARENA])

def expert_action(pos):
    """PD expert controller: proportional drive toward goal, slow near obstacles."""
    diff = GOAL - pos
    d    = np.linalg.norm(diff)
    v    = 3.5 * diff / max(d, 0.01)
    # Obstacle avoidance impulse
    for cx, cy, r in OBSTACLES:
        od = pos - np.array([cx, cy])
        od_d = max(np.linalg.norm(od) - r, 0.01)
        if od_d < 1.5:
            v += 2.0 * (1/od_d - 1/1.5) / od_d * od / np.linalg.norm(od)
    # Clip to max speed
    spd = np.linalg.norm(v)
    if spd > MAX_SPD: v = v / spd * MAX_SPD
    return v

def step_env(pos, action, noise=0.02):
    """Integrate robot position with noise."""
    n = rng.normal(0, noise, 2)
    return np.clip(pos + (action + n) * DT, 0.0, ARENA)

# ─── 1. COLLECT EXPERT DEMONSTRATIONS ─────────────────────────────────────────
print("Isaac GR00T-Style Policy: Behavioural Cloning (NumPy MLP)")
print("=" * 60)
print("Phase 1: Collecting expert demonstrations...")

N_DEMOS    = 20    # number of demo episodes
T_DEMO     = 100   # steps per episode

obs_list, act_list = [], []
demo_paths = []

for ep in range(N_DEMOS):
    pos  = rng.uniform(0.5, 3.0, 2)   # start near bottom-left
    path = [pos.copy()]
    for _ in range(T_DEMO):
        ob  = obs_from_state(pos)
        act = expert_action(pos)
        obs_list.append(ob); act_list.append(act)
        pos = step_env(pos, act)
        path.append(pos.copy())
        if np.linalg.norm(pos - GOAL) < GOAL_R: break
    demo_paths.append(np.array(path))

X_train = np.array(obs_list, dtype=np.float32)
Y_train = np.array(act_list, dtype=np.float32)
print(f"  Collected {len(X_train)} (obs, action) pairs from {N_DEMOS} demos")

# ─── 2. TINY MLP POLICY (3-layer, implemented in NumPy) ──────────────────────
IN_DIM  = X_train.shape[1]    # 5
H       = 48                   # hidden units
OUT_DIM = Y_train.shape[1]    # 2

def relu(z): return np.maximum(0.0, z)
def relu_grad(z): return (z > 0).astype(np.float32)

# Xavier initialisation
W1 = rng.normal(0, np.sqrt(2/IN_DIM), (IN_DIM, H)).astype(np.float32)
b1 = np.zeros((1, H), dtype=np.float32)
W2 = rng.normal(0, np.sqrt(2/H), (H, H)).astype(np.float32)
b2 = np.zeros((1, H), dtype=np.float32)
W3 = rng.normal(0, np.sqrt(2/H), (H, OUT_DIM)).astype(np.float32)
b3 = np.zeros((1, OUT_DIM), dtype=np.float32)

def forward(X):
    z1 = X @ W1 + b1;  a1 = relu(z1)
    z2 = a1 @ W2 + b2; a2 = relu(z2)
    out = a2 @ W3 + b3
    return out, (X, z1, a1, z2, a2)

def backward_update(Y_pred, Y_true, cache, lr=5e-3):
    global W1, b1, W2, b2, W3, b3
    X, z1, a1, z2, a2 = cache
    B = X.shape[0]
    dout = (Y_pred - Y_true) * (2.0/B)
    dW3 = a2.T @ dout;  db3 = dout.sum(0, keepdims=True)
    da2 = dout @ W3.T;  dz2 = da2 * relu_grad(z2)
    dW2 = a1.T @ dz2;   db2 = dz2.sum(0, keepdims=True)
    da1 = dz2 @ W2.T;   dz1 = da1 * relu_grad(z1)
    dW1 = X.T  @ dz1;   db1 = dz1.sum(0, keepdims=True)
    W3 -= lr*dW3; b3 -= lr*db3
    W2 -= lr*dW2; b2 -= lr*db2
    W1 -= lr*dW1; b1 -= lr*db1

# ─── 3. TRAIN ─────────────────────────────────────────────────────────────────
print("Phase 2: Training MLP policy (behavioural cloning)...")
EPOCHS    = 500
BATCH     = 64
N         = len(X_train)
losses    = []
for epoch in range(EPOCHS):
    idx = rng.permutation(N)
    epoch_loss = 0.0
    for start in range(0, N, BATCH):
        Xb = X_train[idx[start:start+BATCH]]
        Yb = Y_train[idx[start:start+BATCH]]
        Ypred, cache = forward(Xb)
        loss = float(np.mean((Ypred - Yb)**2))
        epoch_loss += loss
        backward_update(Ypred, Yb, cache, lr=5e-3)
    losses.append(epoch_loss / max(1, N//BATCH))
    if (epoch+1) % 100 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS}  MSE loss={losses[-1]:.4f}")

# ─── 4. POLICY ROLLOUT ────────────────────────────────────────────────────────
print("Phase 3: Running trained policy (no expert, no PD controller)...")

def policy_action(pos):
    ob = obs_from_state(pos).reshape(1, -1).astype(np.float32)
    action, _ = forward(ob)
    return action[0]

N_ROLLOUTS = 5
policy_paths = []
rollout_results = []
for ep in range(N_ROLLOUTS):
    pos  = rng.uniform(0.5, 3.0, 2)
    path = [pos.copy()]
    reached = False
    for _ in range(T_DEMO):
        act = policy_action(pos)
        pos = step_env(pos, act)
        path.append(pos.copy())
        if np.linalg.norm(pos - GOAL) < GOAL_R:
            reached = True; break
    policy_paths.append(np.array(path))
    rollout_results.append(reached)
    status = "REACHED GOAL" if reached else f"dist={np.linalg.norm(pos-GOAL):.2f}"
    print(f"  Rollout {ep+1}: {status}")

# ─── 5. PLOT ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
ax1, ax2, ax3 = axes
fig.suptitle("GR00T-Style Imitation Learning Policy (NumPy MLP on 2D navigation)",
             fontsize=10)

# Expert demos
ax1.set_xlim(0, ARENA); ax1.set_ylim(0, ARENA); ax1.set_aspect('equal')
ax1.set_facecolor('#f0f0f0'); ax1.set_title("Expert Demonstrations (PD controller)", fontsize=9)
import matplotlib.patches as patches
for cx, cy, r in OBSTACLES:
    ax1.add_patch(patches.Circle((cx,cy), r, facecolor='tomato', alpha=0.7))
for path in demo_paths[:8]:
    ax1.plot(path[:,0], path[:,1], 'b-', alpha=0.3, lw=1)
ax1.plot(*GOAL, 'g*', ms=14, label='Goal')
ax1.legend(fontsize=8); ax1.grid(alpha=0.2)

# Policy rollouts
ax2.set_xlim(0, ARENA); ax2.set_ylim(0, ARENA); ax2.set_aspect('equal')
ax2.set_facecolor('#f0f0f0')
n_ok = sum(rollout_results)
ax2.set_title(f"Policy Rollouts (BC-trained MLP)  {n_ok}/{N_ROLLOUTS} reached goal", fontsize=9)
for cx, cy, r in OBSTACLES:
    ax2.add_patch(patches.Circle((cx,cy), r, facecolor='tomato', alpha=0.7))
colors = ['#2ecc71' if r else '#e74c3c' for r in rollout_results]
for path, col in zip(policy_paths, colors):
    ax2.plot(path[:,0], path[:,1], '-', color=col, alpha=0.7, lw=1.5)
ax2.plot(*GOAL, 'g*', ms=14, label='Goal')
ax2.legend(fontsize=8); ax2.grid(alpha=0.2)

# Training loss
ax3.plot(losses, color='steelblue', lw=1.5)
ax3.set_title("Training Loss (MSE) over 500 Epochs", fontsize=9)
ax3.set_xlabel("Epoch"); ax3.set_ylabel("MSE Loss")
ax3.grid(alpha=0.3); ax3.set_yscale('log')

plt.tight_layout()
plt.savefig("ROBOTICS/ADVANCED_NVIDIA_ROS/outputs/groot_policy.png", dpi=90, bbox_inches='tight')
print("Saved: ROBOTICS/ADVANCED_NVIDIA_ROS/outputs/groot_policy.png")
if INTERACTIVE:
    plt.ioff(); plt.show()
plt.close()
