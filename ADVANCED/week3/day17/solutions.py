# Advanced Day 17 — Solutions
import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import deque

GRID = np.array([
    [0,0,0,1,0,0,0,0,0,0],
    [0,1,0,1,0,1,1,1,0,0],
    [0,1,0,0,0,0,0,1,0,0],
    [0,1,1,1,1,0,0,1,0,0],
    [0,0,0,0,1,0,0,0,0,0],
    [1,1,0,0,1,0,1,1,1,0],
    [0,0,0,0,0,0,0,0,1,0],
    [0,1,1,1,1,1,0,0,1,0],
    [0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
], dtype=int)
ROWS,COLS=GRID.shape; START=(0,0); GOAL=(9,9)
MOVES4=[(-1,0),(1,0),(0,-1),(0,1)]
MOVES8=MOVES4+[(-1,-1),(-1,1),(1,-1),(1,1)]

def in_b(r,c): return 0<=r<ROWS and 0<=c<COLS
def passable(r,c): return in_b(r,c) and GRID[r,c]==0

def reconstruct(cf, n):
    p=[n]
    while n in cf: n=cf[n]; p.append(n)
    return list(reversed(p))

# Exercise 1 — 8-connected A*
print("=== Exercise 1: Diagonal A* ===")
def astar8(start, goal, w=1.0):
    moves=MOVES8; costs={m:1 if m in MOVES4 else np.sqrt(2) for m in moves}
    open_h=[(0.,start)]; came_from={}; g={start:0.}; expanded=0
    while open_h:
        _,cur=heapq.heappop(open_h); expanded+=1
        if cur==goal: return reconstruct(came_from,goal), expanded
        for m in moves:
            nb=(cur[0]+m[0],cur[1]+m[1])
            if not passable(*nb): continue
            tg=g[cur]+costs[m]
            if nb not in g or tg<g[nb]:
                g[nb]=tg
                h=np.sqrt((nb[0]-goal[0])**2+(nb[1]-goal[1])**2)
                heapq.heappush(open_h,(tg+w*h,nb))
                came_from[nb]=cur
    return None, expanded

path4,exp4=astar8(START,GOAL); path8,exp8=astar8(START,GOAL)
print(f"  4-conn A* length={len(path4)}, expanded={exp4}")
print(f"  8-conn A* length={len(path8)}, expanded={exp8}")

# Exercise 2 — Weighted A*
print("\n=== Exercise 2: Weighted A* ===")
def astar_w(start, goal, w=1.0):
    open_h=[(0,start)]; came_from={}; g={start:0}; exp=0
    while open_h:
        _,cur=heapq.heappop(open_h); exp+=1
        if cur==goal: return reconstruct(came_from,goal), exp
        for dr,dc in MOVES4:
            nb=(cur[0]+dr,cur[1]+dc)
            if not passable(*nb): continue
            tg=g[cur]+1
            if nb not in g or tg<g[nb]:
                g[nb]=tg
                h=abs(nb[0]-goal[0])+abs(nb[1]-goal[1])
                heapq.heappush(open_h,(tg+w*h,nb))
                came_from[nb]=cur
    return None, exp

for w in [1.0, 1.5, 2.0]:
    p,e=astar_w(START,GOAL,w)
    print(f"  w={w}: path={len(p) if p else None}  expanded={e}")

# Exercise 3 — Dynamic obstacle
print("\n=== Exercise 3: Dynamic Obstacles ===")
obstacle_path=[(4,5),(4,6),(4,7),(5,7),(6,7),(6,6)]
replans=0
current=(0,0)
total_path=[]
step_count=0
for seg_idx, obs in enumerate(obstacle_path):
    g2=GRID.copy(); g2[obs]=1
    def astar_custom(start,goal,grid):
        def pass2(r,c): return in_b(r,c) and grid[r,c]==0
        open_h=[(0,start)]; cf={}; g_s={start:0}
        while open_h:
            _,cur=heapq.heappop(open_h)
            if cur==goal: return reconstruct(cf,goal)
            for dr,dc in MOVES4:
                nb=(cur[0]+dr,cur[1]+dc)
                if not pass2(*nb): continue
                tg=g_s[cur]+1
                if nb not in g_s or tg<g_s[nb]:
                    g_s[nb]=tg; cf[nb]=cur
                    h=abs(nb[0]-goal[0])+abs(nb[1]-goal[1])
                    heapq.heappush(open_h,(tg+h,nb))
        return None
    p=astar_custom(current,GOAL,g2); replans+=1
    if p:
        steps=min(20,len(p)-1)  # Move 20 steps along path
        current=p[steps]
        total_path+=p[:steps+1]
    step_count+=20
print(f"  Replanned {replans} times, final pos={current}")

# Exercise 4 — Reward shaping
print("\n=== Exercise 4: Reward Shaping ===")
ALPHA,GAMMA,N_EP=0.3,0.95,1500
MOVES_RL=MOVES4

def run_ql(shape=False):
    Q=np.zeros((ROWS,COLS,4))
    rewards=[]
    for ep in range(N_EP):
        eps=max(0.05,1.0-ep*(0.95/N_EP))
        r,c=START; tot=0
        prev_dist=abs(r-GOAL[0])+abs(c-GOAL[1])
        for _ in range(200):
            a=np.random.randint(4) if np.random.rand()<eps else int(np.argmax(Q[r,c]))
            dr,dc=MOVES_RL[a]; nr,nc=r+dr,c+dc
            if not in_b(nr,nc) or GRID[nr,nc]==1: nr,nc=r,c
            done=(nr,nc)==GOAL
            rew=100. if done else -1.
            if shape:
                new_dist=abs(nr-GOAL[0])+abs(nc-GOAL[1])
                rew+=0.1*(prev_dist-new_dist); prev_dist=new_dist
            Q[r,c,a]+=ALPHA*(rew+GAMMA*np.max(Q[nr,nc])-Q[r,c,a])
            r,c=nr,nc; tot+=rew
            if done: break
        rewards.append(tot)
    return rewards

r_plain=run_ql(False); r_shaped=run_ql(True)
w=50
s_plain=np.convolve(r_plain,np.ones(w)/w,mode="valid")
s_shaped=np.convolve(r_shaped,np.ones(w)/w,mode="valid")
plt.figure(figsize=(9,4))
plt.plot(s_plain,label="No shaping"); plt.plot(s_shaped,label="Shaped")
plt.xlabel("Episode"); plt.ylabel("Reward"); plt.legend(); plt.grid(alpha=0.3)
plt.title("Reward Shaping Comparison"); plt.tight_layout()
plt.savefig("reward_shaping.png",dpi=80); plt.close()
print(f"  Plain last-100 avg: {np.mean(r_plain[-100:]):.1f}")
print(f"  Shaped last-100 avg: {np.mean(r_shaped[-100:]):.1f}")
print("  Saved reward_shaping.png")

# Exercise 5 — Value Iteration
print("\n=== Exercise 5: Value Iteration ===")
V=np.full((ROWS,COLS),-np.inf)
V[GRID==1]=np.nan
for r in range(ROWS):
    for c in range(COLS):
        if GRID[r,c]==0: V[r,c]=0.
V[GOAL]=100.
GAMMA_VI=0.95
for _ in range(500):
    V_new=V.copy()
    for r in range(ROWS):
        for c in range(COLS):
            if GRID[r,c]==1: continue
            if (r,c)==GOAL: continue
            vals=[]
            for dr,dc in MOVES4:
                nr,nc=r+dr,c+dc
                if not passable(nr,nc): continue
                vals.append(-1.+GAMMA_VI*V[nr,nc])
            if vals: V_new[r,c]=max(vals)
    if np.nanmax(np.abs(V_new-V))<1e-4: break
    V=V_new

r,c=START; vi_path=[(r,c)]
for _ in range(200):
    best_a,best_v=None,-np.inf
    for a,(dr,dc) in enumerate(MOVES4):
        nr,nc=r+dr,c+dc
        if passable(nr,nc) and V[nr,nc]>best_v: best_v=V[nr,nc]; best_a=a
    if best_a is None: break
    dr,dc=MOVES4[best_a]; r,c=r+dr,c+dc
    vi_path.append((r,c))
    if (r,c)==GOAL: break
print(f"  Value Iteration path length: {len(vi_path)}, reached goal: {vi_path[-1]==GOAL}")
