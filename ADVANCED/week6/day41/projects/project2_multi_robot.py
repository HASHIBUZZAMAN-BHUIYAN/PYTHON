"""
Project: Multi-Robot Coordination System
Teaches: multiple agents on same grid, collision avoidance,
         task assignment, cooperative coverage.
~20 MB RAM, ~1s on CPU
"""
import random, heapq
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque

class MultiRobotEnv:
    def __init__(self, rows=8, cols=8, n_walls=10, seed=42):
        random.seed(seed)
        self.rows=rows; self.cols=cols
        self.walls=set()
        while len(self.walls)<n_walls:
            r,c=random.randint(1,rows-2),random.randint(1,cols-2)
            if (r,c) not in {(0,0),(0,cols-1),(rows-1,0),(rows-1,cols-1)}: self.walls.add((r,c))
        self.dirty=set()
        for _ in range(8):
            r,c=random.randint(0,rows-1),random.randint(0,cols-1)
            if (r,c) not in self.walls: self.dirty.add((r,c))

    def valid(self,r,c): return 0<=r<self.rows and 0<=c<self.cols and (r,c) not in self.walls
    def neighbors(self,r,c):
        for dr,dc in [(-1,0),(1,0),(0,1),(0,-1)]:
            if self.valid(r+dr,c+dc): yield (r+dr,c+dc)

    def bfs_to(self, start, goal):
        if start==goal: return [start]
        q=deque([[start]]); visited={start}
        while q:
            path=q.popleft()
            if path[-1]==goal: return path
            for nb in self.neighbors(*path[-1]):
                if nb not in visited: visited.add(nb); q.append(path+[nb])
        return [start]

class Robot:
    def __init__(self, robot_id, pos, color):
        self.id=robot_id; self.pos=pos; self.color=color
        self.path=[]; self.target=None; self.cleaned=0; self.steps=0

    def assign_target(self, target): self.target=target
    def has_target(self): return self.target is not None

def assign_tasks(robots, dirty_cells):
    available=set(dirty_cells); assignments={}
    for robot in robots:
        if not available: break
        # Assign closest dirty cell
        best=min(available, key=lambda d: abs(d[0]-robot.pos[0])+abs(d[1]-robot.pos[1]))
        assignments[robot.id]=best; available.discard(best)
    return assignments

env=MultiRobotEnv(8,8,10,seed=1)
starts=[(0,0),(0,7),(7,0),(7,7)]
colors=["steelblue","tomato","seagreen","orange"]
robots=[Robot(i,pos,col) for i,(pos,col) in enumerate(zip(starts,colors))]

# Task assignment
dirty_copy=set(env.dirty)
assignments=assign_tasks(robots,dirty_copy)
for r in robots:
    if r.id in assignments:
        r.assign_target(assignments[r.id])
        r.path=env.bfs_to(r.pos, assignments[r.id])

# ─── Simulate ────────────────────────────────────────────────────────────────
print("=== Multi-Robot Coordination ===")
history={r.id:[r.pos] for r in robots}
occupied_next={}

for step in range(25):
    newly_cleaned=set()
    reserved=set()
    for robot in robots:
        if not robot.has_target(): continue
        if len(robot.path)>1:
            next_pos=robot.path[1]
            if next_pos not in reserved:
                reserved.add(next_pos); occupied_next[robot.id]=next_pos
            else:
                occupied_next[robot.id]=robot.pos  # wait
        else:
            occupied_next[robot.id]=robot.pos

    for robot in robots:
        new_pos=occupied_next.get(robot.id,robot.pos)
        if new_pos!=robot.pos:
            robot.path=robot.path[1:]; robot.steps+=1
        robot.pos=new_pos
        history[robot.id].append(robot.pos)
        if robot.pos==robot.target and robot.pos in env.dirty:
            env.dirty.discard(robot.pos); robot.cleaned+=1; newly_cleaned.add(robot.pos)
            robot.target=None; robot.path=[]

    # Reassign
    if newly_cleaned and env.dirty:
        new_assignments=assign_tasks([r for r in robots if not r.has_target()], env.dirty)
        for r in robots:
            if r.id in new_assignments:
                r.assign_target(new_assignments[r.id])
                r.path=env.bfs_to(r.pos,r.target)

for r in robots:
    print(f"  Robot {r.id}: cleaned={r.cleaned}  steps={r.steps}  final={r.pos}")
print(f"  Dirty remaining: {len(env.dirty)}/{len(assignments)+len(env.dirty)}")

# ─── Visualization ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim(-0.5,env.cols-0.5); ax.set_ylim(-0.5,env.rows-0.5)
for r,c in env.walls:
    ax.add_patch(patches.Rectangle((c-0.5,r-0.5),1,1,color="gray"))
orig_dirty=set(); [orig_dirty.update(p) for p in [env.dirty]] # just track remaining
for robot in robots:
    traj=history[robot.id]
    xs=[p[1] for p in traj]; ys=[p[0] for p in traj]
    ax.plot(xs,ys,"-",color=robot.color,alpha=0.5,linewidth=2)
    ax.plot(xs[0],ys[0],"s",color=robot.color,markersize=12,label=f"Robot {robot.id}")
    ax.plot(xs[-1],ys[-1],"*",color=robot.color,markersize=14)
ax.grid(True,alpha=0.3); ax.legend(fontsize=9); ax.set_title("Multi-Robot Cleaning Coverage",fontsize=11)
ax.set_xlabel("Col"); ax.set_ylabel("Row")
plt.tight_layout(); plt.savefig("multi_robot.png",dpi=90); plt.close()
print("Saved multi_robot.png")
