# ROBOTICS Reference — Pathfinding Library
# Provides: astar, bfs, dijkstra, OccupancyGrid

import heapq, math
from collections import deque
import numpy as np


class OccupancyGrid:
    """2D occupancy grid. 0=free, 1=occupied."""
    def __init__(self, rows, cols, default=0):
        self.rows = rows; self.cols = cols
        self.grid = np.full((rows,cols), default, dtype=int)

    def set_obstacle(self, r, c): self.grid[r,c]=1
    def set_free(self, r, c):     self.grid[r,c]=0
    def set_rect(self, r1,c1,r2,c2, val=1):
        self.grid[r1:r2+1, c1:c2+1] = val

    def passable(self, r, c):
        return 0<=r<self.rows and 0<=c<self.cols and self.grid[r,c]==0

    MOVES4 = [(-1,0),(1,0),(0,-1),(0,1)]
    MOVES8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    def neighbors(self, r, c, diagonal=False):
        for dr,dc in (self.MOVES8 if diagonal else self.MOVES4):
            nr,nc=r+dr,c+dc
            if self.passable(nr,nc): yield (nr,nc)

    def inflate(self, radius):
        """Inflate obstacles by radius cells (robot footprint)."""
        from scipy.ndimage import binary_dilation
        struct=np.ones((2*radius+1,2*radius+1))
        self.grid=binary_dilation(self.grid, structure=struct).astype(int)


def _reconstruct(came_from, node):
    path=[node]
    while node in came_from: node=came_from[node]; path.append(node)
    return list(reversed(path))


def astar(grid: OccupancyGrid, start, goal, diagonal=False, heuristic="manhattan"):
    """A* search. Returns path (list of (r,c)) or None."""
    def h(a,b):
        if heuristic=="euclidean": return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
        return abs(a[0]-b[0])+abs(a[1]-b[1])
    diag_cost = math.sqrt(2)
    open_h = [(0., start)]; cf = {}; g = {start: 0.}
    while open_h:
        _, cur = heapq.heappop(open_h)
        if cur == goal: return _reconstruct(cf, goal)
        for nb in grid.neighbors(*cur, diagonal):
            step = diag_cost if (nb[0]-cur[0]!=0 and nb[1]-cur[1]!=0) else 1.
            tg = g[cur] + step
            if nb not in g or tg < g[nb]:
                g[nb]=tg; cf[nb]=cur
                heapq.heappush(open_h, (tg + h(nb,goal), nb))
    return None


def bfs(grid: OccupancyGrid, start, goal):
    """BFS — guaranteed shortest (unweighted) path."""
    queue = deque([start]); visited = {start}; cf = {}
    while queue:
        cur = queue.popleft()
        if cur == goal: return _reconstruct(cf, goal)
        for nb in grid.neighbors(*cur):
            if nb not in visited:
                visited.add(nb); cf[nb]=cur; queue.append(nb)
    return None


def dijkstra(grid: OccupancyGrid, start, goal=None):
    """Dijkstra — returns shortest path to goal, or dist dict to all cells."""
    open_h = [(0., start)]; dist = {start: 0.}; cf = {}
    while open_h:
        d, cur = heapq.heappop(open_h)
        if cur == goal: return _reconstruct(cf, goal)
        for nb in grid.neighbors(*cur):
            nd = d + 1.
            if nb not in dist or nd < dist[nb]:
                dist[nb]=nd; cf[nb]=cur
                heapq.heappush(open_h,(nd,nb))
    return dist if goal is None else None


def smooth_path(path, iterations=3):
    """Simple path smoothing: average neighbor positions."""
    p = list(path)
    for _ in range(iterations):
        for i in range(1, len(p)-1):
            r = (p[i-1][0]+p[i][0]+p[i+1][0])/3
            c = (p[i-1][1]+p[i][1]+p[i+1][1])/3
            p[i] = (r,c)
    return p


# ─── Demo ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    grid = OccupancyGrid(15,15)
    grid.set_rect(3,3,3,10); grid.set_rect(7,4,7,11); grid.set_rect(11,2,11,9)
    start=(0,0); goal=(14,14)

    path_bfs  = bfs(grid, start, goal)
    path_ast4 = astar(grid, start, goal, diagonal=False)
    path_ast8 = astar(grid, start, goal, diagonal=True, heuristic="euclidean")

    print(f"BFS path length   : {len(path_bfs)}")
    print(f"A*(4-conn) length : {len(path_ast4)}")
    print(f"A*(8-conn) length : {len(path_ast8)}")

    fig,axes=plt.subplots(1,3,figsize=(13,5))
    for ax, path, title in zip(axes,
        [path_bfs,path_ast4,path_ast8],
        ["BFS","A*(4-conn)","A*(8-conn)"]):
        display=np.zeros((15,15,3)); display[grid.grid==1]=[0.2,0.2,0.2]
        if path:
            for r,c in path: display[r,c]=[0.2,0.9,0.2]
        display[start]=[0.,0.5,1.]; display[goal]=[1.,0.4,0.]
        ax.imshow(display,interpolation="nearest"); ax.set_title(f"{title} (len={len(path)})")
        ax.axis("off")
    plt.tight_layout(); plt.savefig("pathfinding_demo.png",dpi=80); plt.close()
    print("Saved pathfinding_demo.png")
