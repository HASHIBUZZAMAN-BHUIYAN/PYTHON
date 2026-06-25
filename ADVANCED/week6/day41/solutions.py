# Advanced Day 41 — Solutions
import random, heapq
from collections import deque

EMPTY=0; WALL=1; GOAL=2; DIRT=4

def make_grid(rows,cols,n_walls=10,seed=0):
    random.seed(seed)
    g=[[EMPTY]*cols for _ in range(rows)]
    for _ in range(n_walls):
        r,c=random.randint(1,rows-2),random.randint(1,cols-2)
        g[r][c]=WALL
    g[rows-1][cols-1]=GOAL
    return g

def valid(g,r,c,rows,cols): return 0<=r<rows and 0<=c<cols and g[r][c]!=WALL

def bfs(g,start,goal,rows,cols):
    q=deque([[start]]); visited={start}; explored=0
    while q:
        path=q.popleft(); explored+=1
        if path[-1]==goal: return path,explored
        r,c=path[-1]
        for dr,dc in [(-1,0),(1,0),(0,1),(0,-1)]:
            nr,nc=r+dr,c+dc
            if valid(g,nr,nc,rows,cols) and (nr,nc) not in visited:
                visited.add((nr,nc)); q.append(path+[(nr,nc)])
    return [],explored

# Ex 1
print("=== Ex 1: A* vs BFS ===")
def astar(g,start,goal,rows,cols):
    def h(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])
    pq=[(h(start,goal),0,start,[start])]; visited={}; explored=0
    while pq:
        f,cost,pos,path=heapq.heappop(pq); explored+=1
        if pos in visited: continue
        visited[pos]=cost
        if pos==goal: return path,explored
        r,c=pos
        for dr,dc in [(-1,0),(1,0),(0,1),(0,-1)]:
            nr,nc=r+dr,c+dc
            if valid(g,nr,nc,rows,cols) and (nr,nc) not in visited:
                nc2=cost+1; f2=nc2+h((nr,nc),goal)
                heapq.heappush(pq,(f2,nc2,(nr,nc),path+[(nr,nc)]))
    return [],explored

g=make_grid(8,8,10,seed=5)
bfs_path,bfs_exp=bfs(g,(0,0),(7,7),8,8)
ast_path,ast_exp=astar(g,(0,0),(7,7),8,8)
print(f"  BFS: path={len(bfs_path)} explored={bfs_exp}")
print(f"  A*:  path={len(ast_path)} explored={ast_exp}")
print(f"  A* explored {(ast_exp/max(bfs_exp,1)):.0%} of BFS nodes\n")

# Ex 2
print("=== Ex 2: Sensor Noise ===")
import numpy as np
actual_positions=[(0,0),(1,1),(2,2),(3,3),(4,4)]
def noisy_sensor(pos,std=0.5,seed=None):
    rng=np.random.default_rng(seed)
    nr=int(round(pos[0]+rng.normal(0,std)))
    nc=int(round(pos[1]+rng.normal(0,std)))
    return (max(0,nr),max(0,nc))
def smooth(history,pos,alpha=0.3):
    if not history: return pos
    lr,lc=history[-1]
    return (lr*(1-alpha)+pos[0]*alpha, lc*(1-alpha)+pos[1]*alpha)
history=[]
print(f"  {'Actual':<12}  {'Noisy':<12}  {'Smoothed'}")
for i,pos in enumerate(actual_positions):
    noisy=noisy_sensor(pos,std=0.5,seed=i)
    sm=smooth(history,noisy)
    history.append(sm)
    print(f"  {str(pos):<12}  {str(noisy):<12}  ({sm[0]:.1f},{sm[1]:.1f})")

# Ex 3
print("\n=== Ex 3: Dynamic Obstacles ===")
class SimpleEnv:
    def __init__(self,seed=0):
        random.seed(seed)
        self.rows=self.cols=6; self.walls={(2,2),(3,3),(1,4)}
        self.agent=(0,0); self.goal=(5,5); self.steps=0
        self.obstacles={(4,1),(4,2),(3,5)}
    def move_obstacles(self):
        new_obs=set()
        for r,c in self.obstacles:
            dr,dc=random.choice([(-1,0),(1,0),(0,1),(0,-1)])
            nr,nc=max(0,min(5,r+dr)),max(0,min(5,c+dc))
            new_obs.add((nr,nc) if (nr,nc) not in self.walls else (r,c))
        self.obstacles=new_obs
    def can_move(self,r,c):
        return 0<=r<self.rows and 0<=c<self.cols and (r,c) not in self.walls and (r,c) not in self.obstacles
    def step(self):
        r,c=self.agent; self.move_obstacles()
        for dr,dc,name in [(1,0,"S"),(0,1,"E"),(-1,0,"N"),(0,-1,"W")]:
            nr,nc=r+dr,c+dc
            if self.can_move(nr,nc):
                self.agent=(nr,nc); break
        self.steps+=1; return self.agent==self.goal

results=[]
for ep in range(50):
    e=SimpleEnv(seed=ep)
    reached=False
    for _ in range(30):
        if e.step(): reached=True; break
    results.append(reached)
print(f"  Reached goal: {sum(results)}/50 episodes ({sum(results)/50:.0%})")

# Ex 4
print("\n=== Ex 4: Coverage Agent (Lawnmower) ===")
def boustrophedon_path(rows,cols,walls):
    path=[]; direction=1
    for r in range(rows):
        cols_range=range(cols) if direction>0 else range(cols-1,-1,-1)
        for c in cols_range:
            if (r,c) not in walls: path.append((r,c))
        direction*=-1
    return path
walls={(2,2),(3,3),(1,4),(4,1)}; rows=cols=6
path=boustrophedon_path(rows,cols,walls)
total_non_wall=rows*cols-len(walls)
coverage_50=len(path[:50])
print(f"  Total non-wall cells: {total_non_wall}  Visited in 50 steps: {coverage_50}  Coverage: {coverage_50/total_non_wall:.0%}")

# Ex 5
print("\n=== Ex 5: Multi-objective Reward ===")
class Env2:
    def __init__(self,seed=0):
        random.seed(seed)
        self.rows=self.cols=6
        self.walls={(1,2),(2,4),(3,1),(4,3)}
        self.agent=(0,0); self.goal=(5,5); self.steps=0; self.reward=0
        self.dirt={(0,1),(1,1),(2,2),(3,4)}
    def sense(self):
        r,c=self.agent
        return {"pos":(r,c),"goal":self.goal,"dirt":len(self.dirt),
                "nb":{d:(0<=r+dr<self.rows and 0<=c+dc<self.cols and (r+dr,c+dc) not in self.walls)
                      for d,(dr,dc) in {"N":(-1,0),"S":(1,0),"E":(0,1),"W":(0,-1)}.items()}}
    def act(self,a):
        r,c=self.agent
        deltas={"N":(-1,0),"S":(1,0),"E":(0,1),"W":(0,-1),"CLEAN":(0,0)}
        dr,dc=deltas.get(a,(0,0)); self.steps+=1; self.reward-=2
        if a=="CLEAN":
            if (r,c) in self.dirt: self.dirt.discard((r,c)); self.reward+=5
        else:
            nr,nc=r+dr,c+dc
            if 0<=nr<self.rows and 0<=nc<self.cols and (nr,nc) not in self.walls:
                self.agent=(nr,nc)
            else: self.reward-=20
        if self.agent==self.goal: self.reward+=10
        return self.agent==self.goal

def reactive_policy(s):
    if s["dirt"]>0 and random.random()<0.5: return "CLEAN"
    r,c=s["pos"]; gr,gc=s["goal"]
    if gr>r and s["nb"].get("S"): return "S"
    if gc>c and s["nb"].get("E"): return "E"
    if gr<r and s["nb"].get("N"): return "N"
    return "W"

rewards=[]
for ep in range(3):
    e=Env2(seed=ep)
    for _ in range(25):
        s=e.sense()
        done=e.act(reactive_policy(s))
        if done: break
    rewards.append(e.reward)
    print(f"  Episode {ep+1}: steps={e.steps}  dirt_left={len(e.dirt)}  reward={e.reward}")
print(f"  Mean reward: {sum(rewards)/len(rewards):.1f}")
