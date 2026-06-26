"""
SLAM-Lite: Occupancy Grid Mapping
==================================
What it does:
  Simulates a mobile robot building a 2D map of an unknown environment from
  scratch using ray-casting range sensors (like LiDAR or sonar):
    1. Ground truth: a 22x22 grid maze with walls and corridors
    2. Robot starts in the top-left, executes a random walk that prefers
       cells with lower occupancy-certainty (drives exploration)
    3. Every step, 8 simulated range sensors fire in the cardinal and
       diagonal directions (max range 7 cells)
    4. Cells ALONG each ray are marked free; the FIRST blocked cell is
       marked occupied -- the standard 2D occupancy-grid update rule
  The animation shows ground truth (left) vs. the map being built (right)
  in real time.

What it teaches:
  - Occupancy grid representation: probability-of-occupation per cell
  - Ray casting: how a range sensor converts to a free/occupied update
  - Log-odds update (simplified here as a clamped linear update)
  - Exploration heuristic: bias motion toward uncertain regions
  - Why complete maps require multiple passes and diverse viewpoints

Controls: None -- auto-runs until N_STEPS elapsed.
Animation: Live matplotlib window. Set MPLBACKEND=Agg to skip display (headless).
Output: ROBOTICS/outputs/slam_map.png
RAM: ~60 MB | Time: ~6s with display, <1s headless
"""

import os
import numpy as np
import matplotlib
INTERACTIVE = matplotlib.get_backend().lower() != 'agg'
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

os.makedirs("ROBOTICS/outputs", exist_ok=True)
rng = np.random.default_rng(42)

# ─── WORLD ────────────────────────────────────────────────────────────────────
ROWS, COLS = 22, 22
N_STEPS    = 250
SENSOR_R   = 7      # ray max length (cells)
FREE_DEC   = 0.08   # occupancy decrease for free reading
OCC_INC    = 0.35   # occupancy increase for occupied reading
DRAW_N     = 3

# Build a hand-crafted maze (0=free, 1=wall)
WORLD = np.zeros((ROWS, COLS), dtype=int)
# Border walls
WORLD[0, :] = WORLD[-1, :] = WORLD[:, 0] = WORLD[:, -1] = 1
# Interior walls (create corridors)
for r in range(2, 20, 4):
    for c in range(1, 19):
        if c % 5 != 0:
            WORLD[r, c] = 1
for r in range(4, 20, 4):
    for c in range(2, 21):
        if c % 5 != 2:
            WORLD[r, c] = 1

# Occupancy map: 0.5=unknown, 0.0=known free, 1.0=known occupied
OCC = np.full((ROWS, COLS), 0.5, dtype=float)
# Walls are immediately visible only through sensor, not pre-known
VISITED = np.zeros((ROWS, COLS), dtype=int)  # cells robot has stood on

# ─── RAY CASTING ─────────────────────────────────────────────────────────────
DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]   # 8 directions

def cast_ray(pos, dr, dc, world, occ, max_r):
    """Cast one ray; update occ; return hit position (or None)."""
    r, c = pos
    free_cells, hit = [], None
    for _ in range(1, max_r + 1):
        r += dr; c += dc
        if not (0 <= r < ROWS and 0 <= c < COLS):
            break
        if world[r, c] == 1:   # occupied
            hit = (r, c)
            break
        else:
            free_cells.append((r, c))
    # Log-odds style update (simplified)
    for fr, fc in free_cells:
        OCC[fr, fc] = max(0.0, OCC[fr, fc] - FREE_DEC)
    if hit:
        OCC[hit[0], hit[1]] = min(1.0, OCC[hit[0], hit[1]] + OCC_INC)

def sense_all(pos, world, occ):
    for dr, dc in DIRS:
        cast_ray(pos, dr, dc, world, occ, SENSOR_R)

# ─── ROBOT MOTION ─────────────────────────────────────────────────────────────
def move(pos, world, occ):
    """Move to a random adjacent free cell; prefer uncertain neighbors."""
    r, c = pos
    candidates = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and world[nr, nc] == 0:
                # Prefer cells with high occupancy uncertainty (near 0.5)
                uncertainty = 1.0 - abs(occ[nr, nc] - 0.5) * 2
                candidates.append(((nr, nc), uncertainty + rng.uniform(0, 0.2)))
    if not candidates:
        return pos
    candidates.sort(key=lambda x: -x[1])
    # Choose from top-3 randomly
    top = candidates[:3]
    return rng.choice(len(top))
    choice = top[rng.integers(len(top))]
    return choice[0]

def move_robot(pos, world, occ):
    r, c = pos
    candidates = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and world[nr, nc] == 0:
                uncertainty = 1.0 - abs(occ[nr, nc] - 0.5) * 2
                candidates.append(((nr, nc), uncertainty + rng.uniform(0, 0.25)))
    if not candidates:
        return pos
    candidates.sort(key=lambda x: -x[1])
    top = min(3, len(candidates))
    return candidates[rng.integers(top)][0]

# ─── VISUALIZATION ────────────────────────────────────────────────────────────
print("SLAM-Lite Occupancy Grid Mapping")
print("=" * 45)

robot = (1, 1)
path_cells = [robot]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.5))
fig.suptitle("SLAM-Lite: Occupancy Grid Mapping", fontsize=11)

# Colormap: white=free, grey=unknown, black=wall
cmap_gt  = mcolors.ListedColormap(['white', 'black'])
cmap_occ = plt.colormaps['gray_r']                 # white=free, black=occupied

if INTERACTIVE:
    plt.ion()

def draw(step):
    ax1.clear();  ax2.clear()
    ax1.set_title(f"Ground Truth  (robot sees via sensors only)", fontsize=9)
    ax2.set_title(f"Built Map  step {step}/{N_STEPS}", fontsize=9)

    ax1.imshow(WORLD, cmap=cmap_gt, vmin=0, vmax=1, origin='upper')
    pts = np.array(path_cells)
    ax1.plot(pts[:,1], pts[:,0], 'b-', alpha=0.5, lw=1)
    ax1.plot(robot[1], robot[0], 'ro', ms=8, label='robot')
    ax1.legend(fontsize=8, loc='lower right')

    ax2.imshow(OCC, cmap='RdYlGn_r', vmin=0, vmax=1, origin='upper')
    ax2.plot(pts[:,1], pts[:,0], 'b-', alpha=0.4, lw=1)
    ax2.plot(robot[1], robot[0], 'bo', ms=8)

    # Summary stats
    known_free = np.sum(OCC < 0.3)
    known_occ  = np.sum(OCC > 0.7)
    total_free = np.sum(WORLD == 0)
    ax2.set_xlabel(f"Free cells mapped: {known_free}/{total_free} "
                   f"({100*known_free/max(total_free,1):.0f}%)  "
                   f"Walls found: {known_occ}", fontsize=8)
    plt.tight_layout()

for step in range(N_STEPS):
    sense_all(robot, WORLD, OCC)
    OCC[robot[0], robot[1]] = 0.0    # robot's own cell is definitely free
    VISITED[robot[0], robot[1]] = 1

    robot = move_robot(robot, WORLD, OCC)
    path_cells.append(robot)

    if step % DRAW_N == 0:
        draw(step)
        if INTERACTIVE:
            plt.pause(0.04)

draw(N_STEPS)
known_free = np.sum(OCC < 0.3)
total_free = np.sum(WORLD == 0)
print(f"Mapping complete: {known_free}/{total_free} free cells found "
      f"({100*known_free/total_free:.0f}%)  "
      f"walls identified: {np.sum(OCC > 0.7)}")

plt.savefig("ROBOTICS/outputs/slam_map.png", dpi=90, bbox_inches='tight')
print("Saved: ROBOTICS/outputs/slam_map.png")
if INTERACTIVE:
    plt.ioff();  plt.show()
plt.close()
