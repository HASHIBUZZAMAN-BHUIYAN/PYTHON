# Advanced Day 21 — CAPSTONE: Full Agentic Robotic System
# ~80 MB RAM, ~5s on CPU
# NOTE: heavier lesson — close other apps before running

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import heapq, cv2

print("""
=== Day 21 CAPSTONE: Full Agentic Robotic System ===

Week 3 integration:
  Day 15 — Robot kinematics & simulation
  Day 16 — PID path following
  Day 17 — A* path planning
  Day 18 — OpenCV color beacon detection
  Day 19 — FSM agent controller
  Day 20 — LLM mission planner (mock mode)

System pipeline:
  1. OpenCV detects colored beacons in top-down arena image
  2. LLM agent receives beacon map → decides mission order
  3. FSM controller sequences: DETECT → PLAN → NAVIGATE → ARRIVED
  4. A* plans a discrete grid path (obstacle-aware)
  5. PID steers the robot along the A* waypoints
  6. Visualize full mission trajectory
""")

# ─── 1. ARENA & BEACON SETUP ──────────────────────────────────────────────────
GRID_SIZE = 20       # 20×20 grid cells
CELL_PX   = 20       # pixels per cell for OpenCV rendering
IMG_SIZE  = GRID_SIZE * CELL_PX

# Obstacle map (1=blocked, 0=free)
OBSTACLES = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
WALL_SEGS = [
    ((3,3),(3,8)),((3,3),(8,3)),((8,3),(8,8)),((3,8),(8,8)),    # inner square
    ((11,11),(11,17)),((17,11),(17,17)),((11,17),(17,17)),       # bottom-right box
    ((0,12),(6,12)),((13,0),(13,6)),                             # corridors
]
for (r1,c1),(r2,c2) in WALL_SEGS:
    if r1==r2:
        for c in range(min(c1,c2),max(c1,c2)+1): OBSTACLES[r1,c]=1
    else:
        for r in range(min(r1,r2),max(r1,r2)+1): OBSTACLES[r,c1]=1

# Beacons: (row, col, BGR_color, name)
BEACONS = [
    (1,  18, (0,220,220), "CYAN"),
    (18, 1,  (0,180,0),   "GREEN"),
    (18, 18, (0,0,220),   "RED"),
    (10, 5,  (200,100,0), "ORANGE"),
    (5,  15, (180,0,180), "MAGENTA"),
]
ROBOT_START_GRID = (1, 1)

# ─── 2. OPENCV: DETECT BEACONS ───────────────────────────────────────────────
print("=== 2. OpenCV: Detecting Beacons ===")

def render_arena(obstacles, beacons, robot_pos=None):
    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 200
    # draw obstacles
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if obstacles[r,c]:
                x1,y1 = c*CELL_PX, r*CELL_PX
                cv2.rectangle(img, (x1,y1),(x1+CELL_PX,y1+CELL_PX),(60,60,60),-1)
    # draw beacons
    for r,c,color,name in beacons:
        cx,cy = c*CELL_PX+CELL_PX//2, r*CELL_PX+CELL_PX//2
        cv2.circle(img,(cx,cy),CELL_PX//2-2,color,-1)
    # draw robot
    if robot_pos:
        rr,rc = robot_pos
        cx,cy = rc*CELL_PX+CELL_PX//2, rr*CELL_PX+CELL_PX//2
        cv2.circle(img,(cx,cy),CELL_PX//2-3,(80,0,255),-1)
    return img

arena_img = render_arena(OBSTACLES, BEACONS, ROBOT_START_GRID)
cv2.imwrite("capstone_arena.png", arena_img)

def detect_beacons_cv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ranges = {
        "CYAN":    (np.array([85,100,100]),  np.array([95,255,255])),
        "GREEN":   (np.array([50,100,100]),  np.array([70,255,255])),
        "RED":     (np.array([0,100,100]),   np.array([10,255,255])),
        "ORANGE":  (np.array([10,150,150]),  np.array([25,255,255])),
        "MAGENTA": (np.array([140,100,100]), np.array([160,255,255])),
    }
    found = {}
    for name,(lo,hi) in ranges.items():
        mask = cv2.inRange(hsv,lo,hi)
        cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) < 30: continue
            M = cv2.moments(cnt)
            if M["m00"]==0: continue
            px = int(M["m10"]/M["m00"]); py = int(M["m01"]/M["m00"])
            gc = px//CELL_PX; gr = py//CELL_PX
            found[name] = (gr, gc)
    return found

detected = detect_beacons_cv(arena_img)
print(f"  Detected {len(detected)} beacons:")
for name,(r,c) in detected.items():
    print(f"    {name:8s} at grid ({r},{c})")

# ─── 3. LLM MISSION PLANNER (MOCK) ───────────────────────────────────────────
print("\n=== 3. LLM Mission Planner ===")

class MockMissionLLM:
    def plan_mission(self, beacon_map, robot_pos):
        # Greedy nearest-first mission planning
        remaining = dict(beacon_map)
        pos = robot_pos
        order = []
        while remaining:
            nearest = min(remaining.items(),
                          key=lambda kv: abs(kv[1][0]-pos[0])+abs(kv[1][1]-pos[1]))
            order.append(nearest[0])
            pos = nearest[1]
            del remaining[nearest[0]]
        reasoning = f"Visit beacons nearest-first from {robot_pos}"
        return order, reasoning

llm = MockMissionLLM()
mission_order, reasoning = llm.plan_mission(detected, ROBOT_START_GRID)
print(f"  LLM reasoning: {reasoning}")
print(f"  Mission order: {mission_order}")

# ─── 4. A* PATH PLANNER ──────────────────────────────────────────────────────
MOVES = [(-1,0),(1,0),(0,-1),(0,1)]

def passable(r,c):
    return 0<=r<GRID_SIZE and 0<=c<GRID_SIZE and OBSTACLES[r,c]==0

def astar(start, goal):
    open_h=[(0,start)]; cf={}; g={start:0}
    while open_h:
        _,cur=heapq.heappop(open_h)
        if cur==goal: break
        for dr,dc in MOVES:
            nb=(cur[0]+dr,cur[1]+dc)
            if not passable(*nb): continue
            tg=g[cur]+1
            if nb not in g or tg<g[nb]:
                g[nb]=tg; cf[nb]=cur
                h=abs(nb[0]-goal[0])+abs(nb[1]-goal[1])
                heapq.heappush(open_h,(tg+h,nb))
    if goal not in cf and goal!=start: return None
    p=[goal]
    while p[-1]!=start: p.append(cf[p[-1]])
    return list(reversed(p))

# ─── 5. PID PATH FOLLOWER ────────────────────────────────────────────────────
class PID2D:
    def __init__(self, Kp=2., Ki=0.1, Kd=0.5, dt=0.1, max_f=3.):
        self.Kp=Kp; self.Ki=Ki; self.Kd=Kd; self.dt=dt; self.max_f=max_f
        self.integral=np.zeros(2); self.prev_err=np.zeros(2)

    def step(self, err):
        self.integral=np.clip(self.integral+err*self.dt,-10.,10.)
        D=(err-self.prev_err)/self.dt; self.prev_err=err
        out=self.Kp*err+self.Ki*self.integral+self.Kd*D
        n=np.linalg.norm(out)
        return out/n*self.max_f if n>self.max_f else out

def follow_path(grid_path, start_pos_f):
    pid = PID2D()
    pos = np.array(start_pos_f, dtype=float)
    traj = [pos.copy()]
    for wp in grid_path[1:]:
        goal_f = np.array([wp[0]+0.5, wp[1]+0.5], dtype=float)
        for _ in range(40):
            err = goal_f - pos
            if np.linalg.norm(err) < 0.15: break
            force = pid.step(err)
            pos = pos + force * 0.1
            traj.append(pos.copy())
    return np.array(traj)

# ─── 6. FSM CONTROLLER ───────────────────────────────────────────────────────
print("\n=== 6. Running Mission via FSM Controller ===")

class MissionFSM:
    def __init__(self, mission_order, detected_beacons, start):
        self.queue     = list(mission_order)
        self.beacons   = detected_beacons
        self.pos_grid  = start
        self.pos_f     = np.array([start[0]+0.5, start[1]+0.5], dtype=float)
        self.state     = "PLAN"
        self.full_traj = [self.pos_f.copy()]
        self.segments  = []
        self.arrived   = []

    def run(self):
        while self.queue:
            target_name = self.queue.pop(0)
            if target_name not in self.beacons:
                print(f"  Beacon {target_name} not detected — skip")
                continue
            goal_grid = self.beacons[target_name]

            # PLAN
            path = astar(self.pos_grid, goal_grid)
            if path is None:
                print(f"  No path to {target_name} — skip"); continue
            print(f"  NAVIGATE → {target_name:8s} grid{goal_grid}  path_len={len(path)}")

            # NAVIGATE via PID
            traj = follow_path(path, self.pos_f)
            self.full_traj.extend(traj.tolist())
            self.segments.append((target_name, traj))

            # ARRIVED
            self.pos_grid = goal_grid
            self.pos_f    = traj[-1].copy()
            self.arrived.append(target_name)
            print(f"  ARRIVED   at {target_name}")

        print(f"\n  Mission complete. Visited: {self.arrived}")
        return np.array(self.full_traj)

fsm = MissionFSM(mission_order, detected, ROBOT_START_GRID)
full_traj = fsm.run()

# ─── 7. VISUALIZE FULL MISSION ───────────────────────────────────────────────
print("\n=== 7. Mission Visualization ===")

colors_segs = plt.cm.tab10(np.linspace(0,1,len(fsm.segments)))

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Left: arena overview
ax = axes[0]
# obstacle heatmap
occ = np.zeros((GRID_SIZE,GRID_SIZE))
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        if OBSTACLES[r,c]: occ[r,c]=1
ax.imshow(occ, cmap="Greys", origin="upper", extent=[0,GRID_SIZE,GRID_SIZE,0], vmin=0, vmax=1.5)

# draw trajectory segments
for i,(name,traj) in enumerate(fsm.segments):
    ax.plot(traj[:,1], traj[:,0], color=colors_segs[i], linewidth=2, alpha=0.8)

# beacons
bc_colors = {"CYAN":"cyan","GREEN":"limegreen","RED":"red","ORANGE":"orange","MAGENTA":"magenta"}
for name,(r,c) in detected.items():
    ax.scatter([c+0.5],[r+0.5], s=200, c=bc_colors.get(name,"white"),
               edgecolors="black", linewidths=1.5, zorder=5)
    ax.text(c+0.5, r-0.3, name, ha="center", va="bottom", fontsize=6.5)

# start
ax.scatter([ROBOT_START_GRID[1]+0.5],[ROBOT_START_GRID[0]+0.5],
           s=150, c="purple", marker="^", zorder=6, label="Start")
ax.set_xlim(0,GRID_SIZE); ax.set_ylim(GRID_SIZE,0)
ax.set_title("Full Mission Trajectory"); ax.legend(fontsize=8)
ax.set_xlabel("col"); ax.set_ylabel("row"); ax.grid(alpha=0.2)

# Legend patches
patches = [mpatches.Patch(color=colors_segs[i], label=f"Seg {i+1}: {name}")
           for i,(name,_) in enumerate(fsm.segments)]
ax.legend(handles=patches, fontsize=7, loc="lower right")

# Right: arena top-down image with path overlay
ax2 = axes[1]
vis = cv2.cvtColor(arena_img.copy(), cv2.COLOR_BGR2RGB)
for i,(name,traj) in enumerate(fsm.segments):
    r_arr = (traj[:,0]*CELL_PX).astype(int)
    c_arr = (traj[:,1]*CELL_PX).astype(int)
    col_255 = tuple(int(x*255) for x in colors_segs[i][:3])
    for j in range(1,len(r_arr)):
        cv2.line(vis,(c_arr[j-1],r_arr[j-1]),(c_arr[j],r_arr[j]),col_255,2)
ax2.imshow(vis, interpolation="nearest")
ax2.set_title("Arena Image with Mission Path"); ax2.axis("off")

plt.suptitle("Day 21 CAPSTONE — Agentic Robotic System", fontsize=13)
plt.tight_layout(); plt.savefig("capstone_mission.png", dpi=90)
print("Saved capstone_mission.png")
plt.show()

print("\n" + "="*60)
print("WEEK 3 CAPSTONE COMPLETE")
print("="*60)
print("\nSkills integrated:")
print("  Day 15 — Kinematics")
print("  Day 16 — PID controller")
print("  Day 17 — A* path planning")
print("  Day 18 — OpenCV vision")
print("  Day 19 — FSM agent")
print("  Day 20 — LLM mission planner")
