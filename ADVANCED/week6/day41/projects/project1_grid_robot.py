"""
Project: Grid Navigation Robot with A* and Visualization
Teaches: A* pathfinding, grid environment, step-by-step visualization,
         path comparison (BFS vs A*).
~20 MB RAM, ~1s on CPU
"""
import heapq, random
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

class GridEnv:
    EMPTY=0; WALL=1; GOAL=2; START=3; PATH=4

    def __init__(self, rows=10, cols=10, wall_density=0.2, seed=42):
        random.seed(seed)
        self.rows=rows; self.cols=cols
        self.grid=[[self.EMPTY]*cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if random.random()<wall_density and (r,c) not in [(0,0),(rows-1,cols-1)]:
                    self.grid[r][c]=self.WALL
        self.start=(0,0); self.goal=(rows-1,cols-1)
        self.grid[0][0]=self.START; self.grid[rows-1][cols-1]=self.GOAL

    def neighbors(self, r, c):
        for dr,dc in [(-1,0),(1,0),(0,1),(0,-1)]:
            nr,nc=r+dr,c+dc
            if 0<=nr<self.rows and 0<=nc<self.cols and self.grid[nr][nc]!=self.WALL:
                yield (nr,nc)

def manhattan(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])

def bfs(env):
    from collections import deque
    q=deque([[env.start]]); visited={env.start}; explored=0
    while q:
        path=q.popleft(); explored+=1
        if path[-1]==env.goal: return path,explored
        for nb in env.neighbors(*path[-1]):
            if nb not in visited:
                visited.add(nb); q.append(path+[nb])
    return [],explored

def astar(env):
    start=env.start; goal=env.goal
    pq=[(manhattan(start,goal),0,start,[start])]; visited={}; explored=0
    while pq:
        f,g,pos,path=heapq.heappop(pq); explored+=1
        if pos in visited: continue
        visited[pos]=g
        if pos==goal: return path,explored
        for nb in env.neighbors(*pos):
            if nb not in visited:
                ng=g+1; h=manhattan(nb,goal)
                heapq.heappush(pq,(ng+h,ng,nb,path+[nb]))
    return [],explored

env=GridEnv(10,10,0.25,seed=7)
bfs_path, bfs_exp = bfs(env)
ast_path, ast_exp = astar(env)
print(f"BFS : path length={len(bfs_path)}  nodes explored={bfs_exp}")
print(f"A*  : path length={len(ast_path)}  nodes explored={ast_exp}")
print(f"A* efficiency: explored {ast_exp/max(bfs_exp,1):.0%} of BFS nodes")

# ─── Visualization ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
CMAP = {env.EMPTY:1, env.WALL:0, env.GOAL:0.7, env.START:0.8, env.PATH:0.5}

for ax, path, title in [(axes[0], bfs_path, f"BFS (explore={bfs_exp}, len={len(bfs_path)})"),
                         (axes[1], ast_path, f"A*  (explore={ast_exp}, len={len(ast_path)})")]:
    canvas = np.array([[CMAP.get(cell, 1) for cell in row] for row in env.grid])
    for r,c in path[1:-1]: canvas[r][c] = 0.45
    ax.imshow(canvas, cmap="RdYlGn", vmin=0, vmax=1, origin="upper")
    for r,c in path[1:-1]: ax.plot(c, r, "b.", markersize=4)
    ax.plot(env.start[1], env.start[0], "g^", markersize=10, label="Start")
    ax.plot(env.goal[1],  env.goal[0],  "r*", markersize=12, label="Goal")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8); ax.axis("off")

plt.suptitle("Grid Navigation: BFS vs A*", fontsize=12)
plt.tight_layout(); plt.savefig("grid_robot.png", dpi=90); plt.close()
print("Saved grid_robot.png")
