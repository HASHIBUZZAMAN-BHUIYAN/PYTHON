# Advanced Day 21 Mini-Project — Autonomous Rescue Mission
# Robot navigates a disaster zone, locates survivors (colored beacons),
# and reports findings. Full pipeline: OpenCV → LLM → FSM → A* → PID.
# ~80 MB RAM, ~5s on CPU
# NOTE: heavier lesson — close other apps before running

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import heapq, cv2, math, datetime

print("=== Day 21 Mini-Project: Autonomous Rescue Mission ===\n")

# ─── Environment ─────────────────────────────────────────────────────────────
GRID_SIZE = 25
CELL_PX   = 18
IMG_SIZE  = GRID_SIZE * CELL_PX

OBSTACLES = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
# Building walls (solid blocks)
BUILDINGS = [(2,2,8,8),(2,16,8,22),(16,2,22,8),(16,16,22,22),(10,10,14,14)]
for (r1,c1,r2,c2) in BUILDINGS:
    for r in range(r1,r2+1):
        for c in range(c1,c2+1):
            OBSTACLES[r,c] = 1
# Rubble / barriers (horizontal/vertical lines)
for c in range(9,16): OBSTACLES[11,c] = 1
for r in range(9,16): OBSTACLES[r,11] = 1
for r in range(9,16): OBSTACLES[r,13] = 1
# Clear doorways
OBSTACLES[11,12]=0; OBSTACLES[5,8]=0; OBSTACLES[5,16]=0
OBSTACLES[19,8]=0; OBSTACLES[19,16]=0; OBSTACLES[8,12]=0; OBSTACLES[16,12]=0

# Survivor beacons (yellow=priority, green=low, cyan=medium)
SURVIVORS = [
    (1,  12, (0,220,220), "SURVIVOR_A", 2),   # row,col,BGR,name,priority(1=high)
    (12, 1,  (0,200,0),   "SURVIVOR_B", 3),
    (23, 12, (0,0,220),   "SURVIVOR_C", 1),   # highest priority
    (12, 23, (200,150,0), "SURVIVOR_D", 2),
    (1,  1,  (0,220,220), "SURVIVOR_E", 3),
]
ROBOT_START = (12, 12)
BASE_CAMP   = (0, 0)

# ─── Render arena ─────────────────────────────────────────────────────────────
def render(obstacles, survivors, robot_pos=None):
    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 215
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if obstacles[r,c]:
                x1,y1=c*CELL_PX,r*CELL_PX
                cv2.rectangle(img,(x1,y1),(x1+CELL_PX,y1+CELL_PX),(55,55,55),-1)
    for r,c,col,name,_ in survivors:
        cx,cy=c*CELL_PX+CELL_PX//2, r*CELL_PX+CELL_PX//2
        cv2.circle(img,(cx,cy),CELL_PX//2-1,col,-1)
        cv2.putText(img,name[-1],(cx-4,cy+5),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)
    if robot_pos:
        rr,rc=robot_pos; cx,cy=rc*CELL_PX+CELL_PX//2,rr*CELL_PX+CELL_PX//2
        cv2.circle(img,(cx,cy),CELL_PX//2-2,(120,0,255),-1)
        cv2.circle(img,(cx,cy),CELL_PX//2-2,(255,255,255),1)
    # base camp
    bc=BASE_CAMP; cv2.rectangle(img,(bc[1]*CELL_PX,bc[0]*CELL_PX),
                                 (bc[1]*CELL_PX+CELL_PX,bc[0]*CELL_PX+CELL_PX),(0,120,255),-1)
    return img

arena_img = render(OBSTACLES, SURVIVORS, ROBOT_START)
cv2.imwrite("rescue_arena.png", arena_img)
print("  Saved rescue_arena.png")

# ─── OpenCV: detect survivors ────────────────────────────────────────────────
def detect_survivors(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ranges = {
        "CYAN":   (np.array([85,100,100]), np.array([95,255,255])),
        "GREEN":  (np.array([50,100,100]), np.array([70,255,255])),
        "RED":    (np.array([0,100,100]),  np.array([10,255,255])),
        "ORANGE": (np.array([10,100,100]), np.array([25,255,255])),
    }
    found = {}
    for color,(lo,hi) in ranges.items():
        mask=cv2.inRange(hsv,lo,hi)
        cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for i,cnt in enumerate(cnts):
            if cv2.contourArea(cnt)<25: continue
            M=cv2.moments(cnt)
            if M["m00"]==0: continue
            px=int(M["m10"]/M["m00"]); py=int(M["m01"]/M["m00"])
            key=f"{color}_{i}"; found[key]=((py//CELL_PX),(px//CELL_PX))
    return found

raw_detected = detect_survivors(arena_img)
# Match to known survivors by position
def match_survivors(raw, survivors):
    matched={}
    for r,c,_,name,prio in survivors:
        best_key=min(raw.keys(),key=lambda k:abs(raw[k][0]-r)+abs(raw[k][1]-c),default=None)
        if best_key and abs(raw[best_key][0]-r)+abs(raw[best_key][1]-c)<=2:
            matched[name]=(r,c,prio)
    return matched

matched = match_survivors(raw_detected, SURVIVORS)
print(f"\n  Detected {len(matched)} survivors:")
for name,(r,c,p) in sorted(matched.items(), key=lambda x:x[1][2]):
    print(f"    {name} at ({r},{c}) priority={p}")

# ─── LLM Mission Planner ─────────────────────────────────────────────────────
class RescueLLM:
    def plan(self, survivors, robot_pos):
        # Priority-first, then nearest tie-break
        ranked = sorted(survivors.items(), key=lambda x: (x[1][2], abs(x[1][0]-robot_pos[0])+abs(x[1][1]-robot_pos[1])))
        order  = [name for name,_ in ranked]
        order.append("BASE_CAMP")
        print(f"\n  LLM Mission Plan: {order}")
        return order

llm = RescueLLM()
matched["BASE_CAMP"] = (BASE_CAMP[0], BASE_CAMP[1], 99)
mission = llm.plan(matched, ROBOT_START)

# ─── A* ───────────────────────────────────────────────────────────────────────
def passable(r,c):
    return 0<=r<GRID_SIZE and 0<=c<GRID_SIZE and OBSTACLES[r,c]==0

def astar(start,goal):
    open_h=[(0,start)]; cf={}; g={start:0}
    while open_h:
        _,cur=heapq.heappop(open_h)
        if cur==goal: break
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb=(cur[0]+dr,cur[1]+dc)
            if not passable(*nb): continue
            tg=g[cur]+1
            if nb not in g or tg<g[nb]:
                g[nb]=tg; cf[nb]=cur
                heapq.heappush(open_h,(tg+abs(nb[0]-goal[0])+abs(nb[1]-goal[1]),nb))
    if goal not in cf and goal!=start: return None
    p=[goal]
    while p[-1]!=start: p.append(cf[p[-1]])
    return list(reversed(p))

# ─── PID follower ─────────────────────────────────────────────────────────────
class PID2D:
    def __init__(self, Kp=2.,Ki=0.1,Kd=0.5,dt=0.1,max_f=2.5):
        self.Kp=Kp;self.Ki=Ki;self.Kd=Kd;self.dt=dt;self.max_f=max_f
        self.I=np.zeros(2);self.pe=np.zeros(2)
    def step(self,err):
        self.I=np.clip(self.I+err*self.dt,-10.,10.)
        D=(err-self.pe)/self.dt;self.pe=err
        out=self.Kp*err+self.Ki*self.I+self.Kd*D
        n=np.linalg.norm(out)
        return out/n*self.max_f if n>self.max_f else out

def follow_path(grid_path, start_f):
    pid=PID2D(); pos=np.array(start_f,dtype=float); traj=[pos.copy()]
    for wp in grid_path[1:]:
        goal_f=np.array([wp[0]+0.5,wp[1]+0.5],dtype=float)
        for _ in range(50):
            err=goal_f-pos
            if np.linalg.norm(err)<0.15: break
            pos=pos+pid.step(err)*0.1
            traj.append(pos.copy())
    return np.array(traj)

# ─── FSM Mission Runner ───────────────────────────────────────────────────────
print("\n  Running rescue mission:")
pos_grid = ROBOT_START
pos_f    = np.array([ROBOT_START[0]+0.5, ROBOT_START[1]+0.5], dtype=float)
segments = []
rescue_log = []
start_t = datetime.datetime.now()
total_dist = 0.

for name in mission:
    if name not in matched: continue
    r,c,prio = matched[name]
    path = astar(pos_grid,(r,c))
    if path is None:
        print(f"    SKIP {name}: no path")
        continue
    traj = follow_path(path, pos_f)
    dist = sum(np.linalg.norm(traj[i]-traj[i-1]) for i in range(1,len(traj)))
    total_dist += dist
    segments.append((name, traj, prio))
    pos_grid=(r,c); pos_f=traj[-1].copy()
    status = "RESCUED" if name!="BASE_CAMP" else "RETURNED"
    print(f"    {status}: {name} at ({r},{c})  path={len(path)} cells  dist={dist:.1f}")
    rescue_log.append({"name":name,"pos":(r,c),"dist":dist,"status":status})

end_t = datetime.datetime.now()
print(f"\n  Mission duration: {(end_t-start_t).total_seconds():.2f}s (sim)")
print(f"  Total distance  : {total_dist:.1f} cells")
print(f"  Survivors rescued: {sum(1 for x in rescue_log if x['status']=='RESCUED')}/{len(SURVIVORS)}")

# ─── Visualization ────────────────────────────────────────────────────────────
colors_map = plt.cm.Set1(np.linspace(0,1,len(segments)))
fig, axes = plt.subplots(1,2,figsize=(16,7))

# Grid map
ax=axes[0]
ax.imshow(OBSTACLES,cmap="Greys",origin="upper",extent=[0,GRID_SIZE,GRID_SIZE,0],vmin=0,vmax=2)
for i,(name,traj,prio) in enumerate(segments):
    ax.plot(traj[:,1],traj[:,0],color=colors_map[i],linewidth=2,alpha=0.85,label=f"{name}(p={prio})")
for r,c,_,name,prio in SURVIVORS:
    ax.scatter([c+0.5],[r+0.5],s=180,zorder=5,
               c="yellow" if prio==1 else ("lime" if prio==2 else "cyan"),edgecolors="k",linewidths=1)
    ax.text(c+0.5,r-0.4,name[-1],ha="center",fontsize=7)
ax.scatter([ROBOT_START[1]+0.5],[ROBOT_START[0]+0.5],s=200,c="purple",marker="^",zorder=6,label="Start")
ax.scatter([BASE_CAMP[1]+0.5],[BASE_CAMP[0]+0.5],s=200,c="blue",marker="s",zorder=6,label="Base")
ax.legend(fontsize=7,loc="lower right"); ax.set_title("Rescue Mission — Grid Map")
ax.set_xlabel("col"); ax.set_ylabel("row"); ax.grid(alpha=0.2)
ax.set_xlim(0,GRID_SIZE); ax.set_ylim(GRID_SIZE,0)

# Arena image
ax2=axes[1]
vis=cv2.cvtColor(arena_img.copy(),cv2.COLOR_BGR2RGB)
for i,(name,traj,_) in enumerate(segments):
    c255=tuple(int(x*255) for x in colors_map[i][:3])
    ra=(traj[:,0]*CELL_PX).astype(int); ca=(traj[:,1]*CELL_PX).astype(int)
    for j in range(1,len(ra)): cv2.line(vis,(ca[j-1],ra[j-1]),(ca[j],ra[j]),c255,2)
ax2.imshow(vis,interpolation="nearest")
ax2.set_title("Rescue Mission — Camera View"); ax2.axis("off")

plt.suptitle("Day 21 Mini-Project — Autonomous Rescue Mission", fontsize=13)
plt.tight_layout(); plt.savefig("rescue_mission.png",dpi=90)
print("\nSaved rescue_mission.png")

# ─── Mission Report ───────────────────────────────────────────────────────────
report=["=== RESCUE MISSION REPORT ===",
        f"Date: {start_t.strftime('%Y-%m-%d %H:%M')}",
        f"Robot: RESCUE-1  Start: {ROBOT_START}",
        "","Survivor Status:"]
for x in rescue_log:
    if x["name"]=="BASE_CAMP": continue
    report.append(f"  [{x['status']:>7}] {x['name']} at {x['pos']}  dist={x['dist']:.1f}")
report+=[f"","Survivors rescued: {sum(1 for x in rescue_log if x['status']=='RESCUED')}/{len(SURVIVORS)}",
         f"Total path distance: {total_dist:.1f} cells","Mission status: COMPLETE"]
with open("rescue_report.txt","w") as f: f.write("\n".join(report)+"\n")
print("Saved rescue_report.txt")
plt.show()
