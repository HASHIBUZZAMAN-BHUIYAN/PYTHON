# Advanced Day 21 — Solutions
import numpy as np, heapq, math, datetime, os

GRID_SIZE=20
CELL_PX=20
OBSTACLES = np.zeros((GRID_SIZE,GRID_SIZE),dtype=int)
WALL_SEGS=[((3,3),(3,8)),((3,3),(8,3)),((8,3),(8,8)),((3,8),(8,8)),
           ((11,11),(11,17)),((17,11),(17,17)),((11,17),(17,17)),
           ((0,12),(6,12)),((13,0),(13,6))]
for (r1,c1),(r2,c2) in WALL_SEGS:
    if r1==r2:
        for c in range(min(c1,c2),max(c1,c2)+1): OBSTACLES[r1,c]=1
    else:
        for r in range(min(r1,r2),max(r1,r2)+1): OBSTACLES[r,c1]=1
BEACONS=[(1,18,"CYAN"),(18,1,"GREEN"),(18,18,"RED"),(10,5,"ORANGE"),(5,15,"MAGENTA")]
DETECTED={n:(r,c) for r,c,n in BEACONS}
START=(1,1)

def passable(r,c,obs=None):
    if obs is None: obs=OBSTACLES
    return 0<=r<GRID_SIZE and 0<=c<GRID_SIZE and obs[r,c]==0

def astar_gen(start,goal,obs=None,moves8=False):
    if obs is None: obs=OBSTACLES
    moves=[(-1,0),(1,0),(0,-1),(0,1)]
    if moves8: moves+=[((-1,-1),(-1,1),(1,-1),(1,1))]; moves=moves[:4]+[(-1,-1),(-1,1),(1,-1),(1,1)]
    costs={m:(math.sqrt(2) if abs(m[0])+abs(m[1])==2 else 1.) for m in moves}
    open_h=[(0.,start)]; cf={}; g={start:0.}
    while open_h:
        _,cur=heapq.heappop(open_h)
        if cur==goal: break
        for m in moves:
            nb=(cur[0]+m[0],cur[1]+m[1])
            if not passable(*nb,obs): continue
            tg=g[cur]+costs[m]
            if nb not in g or tg<g[nb]:
                g[nb]=tg; cf[nb]=cur
                h=abs(nb[0]-goal[0])+abs(nb[1]-goal[1])
                heapq.heappush(open_h,(tg+h,nb))
    if goal not in cf and goal!=start: return None
    p=[goal]
    while p[-1]!=start: p.append(cf[p[-1]])
    return list(reversed(p))

# Exercise 1 — Diagonal A*
print("=== Exercise 1: Diagonal A* ===")
mission_order=["CYAN","GREEN","RED","ORANGE","MAGENTA"]
for use8 in [False,True]:
    total=0; prev=START
    for name in mission_order:
        goal=DETECTED[name]
        p=astar_gen(prev,goal,moves8=use8)
        total+=len(p)-1 if p else 999
        prev=goal
    print(f"  {'8-connected' if use8 else '4-connected'}: total steps = {total}")

# Exercise 2 — Dynamic obstacle replanning
print("\n=== Exercise 2: Dynamic Obstacle Replanning ===")
obs_dyn=OBSTACLES.copy()
visited=0; prev=START; replans=0
for name in mission_order:
    goal=DETECTED[name]
    p=astar_gen(prev,goal,obs=obs_dyn)
    if p is None: print(f"  No path to {name}"); continue
    print(f"  Navigating to {name} (len={len(p)})")
    visited+=1
    prev=goal
    if visited==2:   # add wall after 2nd beacon
        obs_dyn[8,8]=1; obs_dyn[9,8]=1; obs_dyn[10,8]=1
        print("  ! Dynamic obstacle added at (8-10, 8)")
        p2=astar_gen(prev,DETECTED[mission_order[2]],obs=obs_dyn)
        if p2: replans+=1; print(f"  Replanned path to {mission_order[2]} (len={len(p2)})")
        else: print(f"  No replan path found")
print(f"  Replanning calls: {replans}")

# Exercise 3 — Battery-aware
print("\n=== Exercise 3: Battery-Aware Mission ===")
CHARGER=(10,10)
battery=100.; battery_log=[battery]; pos=START
def drain_per_path(path_len): return path_len*1.
def charge_to_full(pos):
    p=astar_gen(pos,CHARGER)
    if p: return len(p)*1.,CHARGER
    return 0.,pos
for name in mission_order:
    goal=DETECTED[name]
    p=astar_gen(pos,goal)
    if p is None: continue
    est_drain=drain_per_path(len(p))
    if battery-est_drain<25:
        drain_to_c,new_pos=charge_to_full(pos)
        battery=max(0.,battery-drain_to_c)
        battery+=min(50.,100.-battery)  # charge up
        print(f"  Recharging at {CHARGER} (battery now {battery:.0f})")
        pos=new_pos
        p=astar_gen(pos,goal)
        if p is None: continue
    battery=max(0.,battery-drain_per_path(len(p)))
    battery_log.append(battery); pos=goal
    print(f"  Reached {name}, battery={battery:.0f}")

# Exercise 4 — Multi-robot
print("\n=== Exercise 4: Multi-Robot ===")
robot_a_goals=["CYAN","RED","MAGENTA"]
robot_b_goals=["GREEN","ORANGE"]
def run_robot(goals,start,label):
    pos=start; total=0
    for name in goals:
        goal=DETECTED[name]
        p=astar_gen(pos,goal)
        if p:
            total+=len(p)
            print(f"  Robot {label}: {name} (path_len={len(p)})")
            pos=goal
    print(f"  Robot {label} total steps: {total}")
run_robot(robot_a_goals,START,"A")
run_robot(robot_b_goals,(18,18),"B")

# Exercise 5 — Mission report
print("\n=== Exercise 5: Mission Report ===")
start_time=datetime.datetime.now()
report_lines=[]
report_lines.append("=== MISSION REPORT ===")
report_lines.append(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
total_dist=0.; pos=np.array([START[0]+0.5,START[1]+0.5],dtype=float)
arrival_times=[]
for i,name in enumerate(mission_order):
    goal=DETECTED[name]; gf=np.array([goal[0]+0.5,goal[1]+0.5],dtype=float)
    p=astar_gen(tuple(pos.astype(int)),goal)
    if p is None:
        report_lines.append(f"  SKIP {name}: no path"); continue
    dist=sum(math.sqrt((p[j][0]-p[j-1][0])**2+(p[j][1]-p[j-1][1])**2) for j in range(1,len(p)))
    total_dist+=dist; pos=gf
    t=start_time+datetime.timedelta(seconds=total_dist*0.5)
    arrival_times.append((name,t))
    report_lines.append(f"  Beacon {i+1}: {name} arrived ~{t.strftime('%H:%M:%S')}")
end_time=start_time+datetime.timedelta(seconds=total_dist*0.5)
report_lines.append(f"End:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append(f"Total distance: {total_dist:.1f} cells")
duration=(end_time-start_time).total_seconds()
report_lines.append(f"Duration: {duration:.1f}s")
report_lines.append(f"Avg speed: {total_dist/max(duration,1):.2f} cells/s")
report_lines.append(f"Beacons visited: {len(arrival_times)}/{len(mission_order)}")
report_text="\n".join(report_lines)
print(report_text)
with open("mission_report.txt","w") as f: f.write(report_text+"\n")
print("\nSaved mission_report.txt")
