# Advanced Day 17 — Path Planning: BFS, A*, Q-Learning
# ~35 MB RAM, <3s on CPU

import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import deque

# ─── 1. CONCEPTS ─────────────────────────────────────────────────────────────
print("""
=== Path Planning ===

Goal: find a sequence of moves from START to GOAL on a grid, avoiding obstacles.

  BFS  — explores all nodes at distance d before d+1 (uninformed, guaranteed shortest)
  A*   — uses f(n) = g(n) + h(n), where h is a heuristic (Manhattan/Euclidean)
         Much faster than BFS when the heuristic is good.
  Q-learning — model-free RL; agent learns which action to take by trial and error.

Grid conventions:
  0 = free cell
  1 = obstacle
  S = start, G = goal
  Moves: up, down, left, right (4-connected)
""")

# ─── 2. GRID DEFINITION ───────────────────────────────────────────────────────
GRID = np.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

START = (0, 0)
GOAL  = (9, 9)
ROWS, COLS = GRID.shape
MOVES = [(-1,0),(1,0),(0,-1),(0,1)]  # up, down, left, right

def in_bounds(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS

def passable(r, c):
    return in_bounds(r, c) and GRID[r, c] == 0

def neighbors(r, c):
    return [(r+dr, c+dc) for dr,dc in MOVES if passable(r+dr, c+dc)]

def reconstruct(came_from, node):
    path = [node]
    while node in came_from:
        node = came_from[node]
        path.append(node)
    return list(reversed(path))

# ─── 3. BFS ──────────────────────────────────────────────────────────────────
print("=== 3. BFS ===")

def bfs(grid, start, goal):
    queue = deque([start])
    visited = {start}
    came_from = {}
    nodes_expanded = 0
    while queue:
        node = queue.popleft()
        nodes_expanded += 1
        if node == goal:
            return reconstruct(came_from, goal), nodes_expanded
        for nb in neighbors(*node):
            if nb not in visited:
                visited.add(nb)
                came_from[nb] = node
                queue.append(nb)
    return None, nodes_expanded

bfs_path, bfs_expanded = bfs(GRID, START, GOAL)
print(f"  BFS path length  : {len(bfs_path) if bfs_path else 'No path'}")
print(f"  BFS nodes expanded: {bfs_expanded}")

# ─── 4. A* ───────────────────────────────────────────────────────────────────
print("\n=== 4. A* ===")

def heuristic(a, b, h_type="manhattan"):
    if h_type == "manhattan":
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def astar(grid, start, goal, h_type="manhattan"):
    open_heap = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    nodes_expanded = 0
    while open_heap:
        _, current = heapq.heappop(open_heap)
        nodes_expanded += 1
        if current == goal:
            return reconstruct(came_from, goal), nodes_expanded
        for nb in neighbors(*current):
            tentative_g = g_score[current] + 1
            if nb not in g_score or tentative_g < g_score[nb]:
                g_score[nb] = tentative_g
                f = tentative_g + heuristic(nb, goal, h_type)
                heapq.heappush(open_heap, (f, nb))
                came_from[nb] = current
    return None, nodes_expanded

astar_path, astar_expanded = astar(GRID, START, GOAL)
print(f"  A* path length    : {len(astar_path) if astar_path else 'No path'}")
print(f"  A* nodes expanded : {astar_expanded}")
print(f"  BFS expanded {bfs_expanded} nodes vs A* {astar_expanded} — speedup = {bfs_expanded/max(astar_expanded,1):.1f}x")

# ─── 5. VISUALIZE BFS vs A* ───────────────────────────────────────────────────
def plot_grid(ax, grid, path, title, path_color="lime"):
    display = np.zeros((*grid.shape, 3))
    display[grid == 1] = [0.2, 0.2, 0.2]     # obstacle = dark
    if path:
        for r,c in path:
            display[r,c] = [0.0, 0.8, 0.0]    # path = green
    r0,c0 = START; rg,cg = GOAL
    display[r0,c0] = [0., 0.4, 1.]            # start = blue
    display[rg,cg] = [1., 0.4, 0.]            # goal = orange
    ax.imshow(display, interpolation="nearest")
    ax.set_title(title); ax.axis("off")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
plot_grid(axes[0], GRID, bfs_path,   f"BFS  (path={len(bfs_path)}, expanded={bfs_expanded})")
plot_grid(axes[1], GRID, astar_path, f"A*   (path={len(astar_path)}, expanded={astar_expanded})")
plt.suptitle("Path Planning — BFS vs A*", fontsize=12)
plt.tight_layout(); plt.savefig("pathfinding.png", dpi=90)
print("\nSaved pathfinding.png")

# ─── 6. Q-LEARNING GRID WORLD ────────────────────────────────────────────────
print("\n=== 6. Q-Learning Grid World ===")
print("""
Model-free RL: agent discovers the path by exploration.
  State  = (row, col)
  Actions= 0=up, 1=down, 2=left, 3=right
  Reward = +100 at goal, -1 per step, -10 hitting obstacle (stays in place)

Update: Q[s,a] += alpha * (r + gamma*max(Q[s']) - Q[s,a])
""")

ALPHA = 0.3
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END   = 0.05
N_EPISODES    = 2000

Q = np.zeros((ROWS, COLS, 4))

def step_env(r, c, action):
    dr, dc = MOVES[action]
    nr, nc = r+dr, c+dc
    if not in_bounds(nr, nc) or GRID[nr, nc] == 1:
        return r, c, -10., False      # bounce back
    if (nr, nc) == GOAL:
        return nr, nc, 100., True
    return nr, nc, -1., False

rewards_per_ep = []
for ep in range(N_EPISODES):
    epsilon = max(EPSILON_END, EPSILON_START - ep*(EPSILON_START-EPSILON_END)/N_EPISODES)
    r, c = START
    total_r = 0
    for _ in range(200):                # max steps per episode
        if np.random.rand() < epsilon:
            a = np.random.randint(4)
        else:
            a = int(np.argmax(Q[r, c]))
        nr, nc, reward, done = step_env(r, c, a)
        best_next = np.max(Q[nr, nc])
        Q[r, c, a] += ALPHA * (reward + GAMMA * best_next - Q[r, c, a])
        r, c = nr, nc
        total_r += reward
        if done:
            break
    rewards_per_ep.append(total_r)

print(f"  Training done. Last 100 eps avg reward = {np.mean(rewards_per_ep[-100:]):.1f}")

# Extract greedy policy path
def greedy_path(Q, start, goal, max_steps=200):
    r, c = start
    path = [(r, c)]
    visited = {(r, c)}
    for _ in range(max_steps):
        a = int(np.argmax(Q[r, c]))
        dr, dc = MOVES[a]
        nr, nc = r+dr, c+dc
        if not in_bounds(nr, nc) or GRID[nr, nc] == 1:
            break
        r, c = nr, nc
        if (r, c) in visited:
            break
        visited.add((r, c)); path.append((r, c))
        if (r, c) == goal:
            break
    return path

ql_path = greedy_path(Q, START, GOAL)
found = ql_path[-1] == GOAL
print(f"  Q-learning greedy path: {'FOUND goal' if found else 'did not reach goal'}, length={len(ql_path)}")

# ─── 7. VISUALIZE Q-LEARNING ─────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5))

plot_grid(axes2[0], GRID, ql_path,
          f"Q-Learning greedy path ({'goal!' if found else 'partial'})")

# Smoothed reward curve
window = 50
smoothed = np.convolve(rewards_per_ep, np.ones(window)/window, mode="valid")
axes2[1].plot(smoothed, color="purple")
axes2[1].set_title("Q-Learning: Reward per Episode (smoothed)")
axes2[1].set_xlabel("Episode"); axes2[1].set_ylabel("Total reward")
axes2[1].grid(alpha=0.3)

plt.suptitle("Day 17 — Path Planning & Q-Learning", fontsize=12)
plt.tight_layout(); plt.savefig("qlearning.png", dpi=90)
print("Saved qlearning.png")
plt.show()
