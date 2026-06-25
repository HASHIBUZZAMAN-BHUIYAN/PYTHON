# Advanced Day 41 — Agents + Simulation (Robotics-Inspired)
# ~20 MB RAM, <1s on CPU

print("""
=== Agents + Simulation — Day 41 ===

Robotics-inspired agent concepts (simulated, no hardware needed):

  Environment  — the world the agent lives in (grid, map, state machine)
  Sensor       — agent's read-only view of the environment
  Actuator     — how the agent changes the environment
  Policy       — mapping from state → action
  Reward       — scalar signal for agent evaluation

Key robotics-AI patterns:
  Reactive     — direct sense→act loop (no memory)
  Deliberative — plan ahead, then act
  Hybrid       — reactive for safety, deliberative for goals
""")

import random
from collections import deque

# ─── 1. GRID WORLD ─────────────────────────────────────────────────────────
print("=== 1. Grid World Environment ===")

class GridWorld:
    EMPTY=0; WALL=1; GOAL=2; AGENT=3; DIRT=4

    def __init__(self, rows=6, cols=6, seed=42):
        random.seed(seed)
        self.rows=rows; self.cols=cols
        self.grid=[[self.EMPTY]*cols for _ in range(rows)]
        # Walls
        for _ in range(6): self.grid[random.randint(1,rows-2)][random.randint(1,cols-2)]=self.WALL
        # Dirt
        self.dirt_cells=set()
        for _ in range(5):
            r,c=random.randint(0,rows-1),random.randint(0,cols-1)
            if self.grid[r][c]==self.EMPTY: self.grid[r][c]=self.DIRT; self.dirt_cells.add((r,c))
        self.agent_pos=(0,0); self.goal=(rows-1,cols-1)
        self.grid[self.goal[0]][self.goal[1]]=self.GOAL
        self.steps=0; self.reward=0

    def valid(self, r,c): return 0<=r<self.rows and 0<=c<self.cols and self.grid[r][c]!=self.WALL

    def sense(self):
        r,c=self.agent_pos
        return {"pos":(r,c),"goal":self.goal,"dirt":len(self.dirt_cells),
                "neighbors":{d:self.valid(r+dr,c+dc) for d,(dr,dc) in {"N":(-1,0),"S":(1,0),"E":(0,1),"W":(0,-1)}.items()}}

    def act(self, action):
        r,c=self.agent_pos
        deltas={"N":(-1,0),"S":(1,0),"E":(0,1),"W":(0,-1),"CLEAN":(0,0)}
        dr,dc=deltas.get(action,(0,0))
        nr,nc=r+dr,c+dc
        if action=="CLEAN":
            if (r,c) in self.dirt_cells: self.dirt_cells.discard((r,c)); self.grid[r][c]=self.EMPTY; self.reward+=10
        elif self.valid(nr,nc):
            self.agent_pos=(nr,nc)
        self.steps+=1; self.reward-=1
        return self.sense()

    def render(self, max_rows=6):
        SYMS={self.EMPTY:".",self.WALL:"#",self.GOAL:"G",self.DIRT:"*"}
        grid_copy=[row[:] for row in self.grid]
        ar,ac=self.agent_pos; grid_copy[ar][ac]=self.AGENT
        for row in grid_copy[:max_rows]:
            print("  "+"".join("A" if v==self.AGENT else SYMS.get(v,"?") for v in row))

env=GridWorld(6,6)
print("  Initial world (A=agent, #=wall, *=dirt, G=goal):")
env.render()
sense=env.sense()
print(f"  Agent at {sense['pos']}, dirt cells remaining: {sense['dirt']}, goal: {sense['goal']}\n")

# ─── 2. REACTIVE AGENT ─────────────────────────────────────────────────────
print("=== 2. Reactive Robot Agent ===")

class ReactiveAgent:
    def __init__(self, name): self.name=name
    def choose_action(self, sense_data):
        pos=sense_data["pos"]; goal=sense_data["goal"]; nb=sense_data["neighbors"]
        gr,gc=goal; ar,ac=pos
        # Clean if on dirt
        if sense_data["dirt"] > 0: return "CLEAN"
        # Move toward goal
        if gr>ar and nb.get("S"): return "S"
        if gc>ac and nb.get("E"): return "E"
        if gr<ar and nb.get("N"): return "N"
        if gc<ac and nb.get("W"): return "W"
        # Fallback
        for d in ["S","E","N","W"]:
            if nb.get(d): return d
        return "S"

env2=GridWorld(6,6); agent=ReactiveAgent("Robot-1")
print(f"  Running {agent.name} for 15 steps...")
for step in range(15):
    s=env2.sense(); a=agent.choose_action(s)
    env2.act(a)
print(f"  Final pos: {env2.agent_pos}  Dirt left: {len(env2.dirt_cells)}  Steps: {env2.steps}  Reward: {env2.reward}")

# ─── 3. BFS DELIBERATIVE AGENT ─────────────────────────────────────────────
print("\n=== 3. BFS Deliberative Agent ===")

def bfs_path(grid, start, goal, rows, cols):
    WALL=1; q=deque([[start]]); visited={start}
    while q:
        path=q.popleft()
        if path[-1]==goal: return path
        r,c=path[-1]
        for dr,dc in [(-1,0),(1,0),(0,1),(0,-1)]:
            nr,nc=r+dr,c+dc
            if 0<=nr<rows and 0<=nc<cols and grid[nr][nc]!=WALL and (nr,nc) not in visited:
                visited.add((nr,nc)); q.append(path+[(nr,nc)])
    return []

env3=GridWorld(6,6)
path=bfs_path(env3.grid,env3.agent_pos,env3.goal,env3.rows,env3.cols)
print(f"  BFS path length: {len(path)} steps  Path: {path[:5]}...")
for tgt in path[1:]:
    cr,cc=env3.agent_pos; tr,tc=tgt
    if tr>cr: env3.act("S")
    elif tr<cr: env3.act("N")
    elif tc>cc: env3.act("E")
    else: env3.act("W")
print(f"  Arrived at: {env3.agent_pos}  Steps: {env3.steps}  Reward: {env3.reward}")

print("""
=== Summary ===
  Reactive    — fast, no memory, handles immediate obstacles
  Deliberative— plans optimal path via BFS/A*, slower to start
  Hybrid      — plan globally, react locally to unexpected obstacles
""")
