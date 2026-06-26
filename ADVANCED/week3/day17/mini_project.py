# Advanced Day 17 Mini-Project — Multi-Room Navigation with A*
# A robot navigates a multi-room floor plan from entrance to target room.
# ~30 MB RAM, <3s on CPU

import numpy as np
import matplotlib.pyplot as plt
import heapq

print("=== Day 17 Mini-Project: Multi-Room Navigation ===\n")

# ─── Floor plan ───────────────────────────────────────────────────────────────
# 0=open, 1=wall
FLOOR = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,1,1,0,1,1,1,1,1,0,1,1,1,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,1,1,0,1,1,1,0,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,1,1,1,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,1,1,0,1,1,1,0,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
], dtype=int)

ROWS,COLS = FLOOR.shape
START = (1,1)    # Room A (entrance)
GOAL  = (18,18)  # Room F (target)
WAYPOINTS = [(3,3),(7,3),(10,10),(15,15)]  # optional intermediate points

# ─── A* ───────────────────────────────────────────────────────────────────────
MOVES = [(-1,0),(1,0),(0,-1),(0,1)]

def passable(r,c):
    return 0<=r<ROWS and 0<=c<COLS and FLOOR[r,c]==0

def reconstruct(cf, n):
    p=[n]
    while n in cf: n=cf[n]; p.append(n)
    return list(reversed(p))

def astar(start, goal):
    h = lambda a,b: abs(a[0]-b[0])+abs(a[1]-b[1])
    open_h=[(0,start)]; cf={}; g={start:0}; expanded=0
    while open_h:
        _,cur=heapq.heappop(open_h); expanded+=1
        if cur==goal: return reconstruct(cf,goal), expanded
        for dr,dc in MOVES:
            nb=(cur[0]+dr,cur[1]+dc)
            if not passable(*nb): continue
            tg=g[cur]+1
            if nb not in g or tg<g[nb]:
                g[nb]=tg; cf[nb]=cur
                heapq.heappush(open_h,(tg+h(nb,goal),nb))
    return None, expanded

# ─── Direct A* ────────────────────────────────────────────────────────────────
direct_path, direct_exp = astar(START, GOAL)
if direct_path:
    print(f"Direct A* (start→goal): length={len(direct_path)}, expanded={direct_exp}")
else:
    print("Direct A*: no path found")

# ─── Via waypoints ────────────────────────────────────────────────────────────
checkpoints = [START] + WAYPOINTS + [GOAL]
waypoint_path = []
total_exp = 0
prev = START
for nxt in checkpoints[1:]:
    seg, exp = astar(prev, nxt)
    if seg is None:
        print(f"  No path from {prev} to {nxt} — skipping waypoint")
        continue
    waypoint_path += seg[:-1]  # avoid duplicating junction
    total_exp += exp
    prev = nxt
waypoint_path.append(GOAL)
print(f"Via waypoints      : length={len(waypoint_path)}, total_expanded={total_exp}")
print(f"Overhead of waypoints: +{len(waypoint_path)-len(direct_path)} steps vs direct")

from collections import deque
def bfs_coverage(start):
    queue=deque([start]); vis={start}; dist={start:0}
    while queue:
        cur=queue.popleft()
        for dr,dc in MOVES:
            nb=(cur[0]+dr,cur[1]+dc)
            if passable(*nb) and nb not in vis:
                vis.add(nb); dist[nb]=dist[cur]+1; queue.append(nb)
    return dist

dist_map = bfs_coverage(START)
D_GRID = np.full((ROWS,COLS), np.nan)
for (r,c),d in dist_map.items():
    D_GRID[r,c] = d

# ─── Visualize ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1,3, figsize=(16,6))

def draw_path(ax, path, title, color):
    display = np.ones((*FLOOR.shape,3))*0.95
    display[FLOOR==1] = [0.15,0.15,0.15]
    if path:
        for r,c in path: display[r,c]=color
    r0,c0=START; rg,cg=GOAL
    display[r0,c0]=[0.,0.5,1.]; display[rg,cg]=[1.,0.4,0.]
    ax.imshow(display, interpolation="nearest")
    ax.set_title(title); ax.axis("off")

draw_path(axes[0], direct_path,   f"Direct A* (len={len(direct_path)})", [0.2,0.9,0.2])
draw_path(axes[1], waypoint_path, f"Via waypoints (len={len(waypoint_path)})", [0.9,0.7,0.1])

axes[2].imshow(D_GRID, cmap="viridis", interpolation="nearest")
axes[2].plot([START[1]],[START[0]],"w^",markersize=12,label="Start")
axes[2].plot([GOAL[1]], [GOAL[0]], "w*", markersize=14, label="Goal")
axes[2].set_title("BFS Distance from Start"); axes[2].axis("off")
axes[2].legend(fontsize=8, loc="lower right")

plt.suptitle("Day 17 Mini-Project — Multi-Room Navigation", fontsize=12)
plt.tight_layout(); plt.savefig("room_navigation.png", dpi=90)
print("\nSaved room_navigation.png")
plt.show()
