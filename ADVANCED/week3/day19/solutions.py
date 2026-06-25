# Advanced Day 19 — Solutions
import random

# Exercise 1 — Priority reflex agent
print("=== Exercise 1: Priority Reflex Agent ===")
class DroneReflex:
    def __init__(self):
        self.battery = 80.
    def step(self, percept):
        self.battery = max(0., self.battery - random.uniform(3,7))
        if self.battery < 20: return "RETURN_TO_BASE"
        if percept.get("obstacle"): return "AVOID_OBSTACLE"
        if percept.get("target"):   return "TRACK_TARGET"
        return "SEARCH_PATTERN"

random.seed(1); drone = DroneReflex()
print(f"{'Step':>4}  {'Battery':>8}  Action")
print("-"*35)
for i in range(15):
    p = {"obstacle": random.random()<0.2, "target": random.random()<0.3}
    a = drone.step(p)
    print(f"{i:>4}  {drone.battery:>8.1f}  {a}")

# Exercise 2 — FSM with battery
print("\n=== Exercise 2: FSM with Battery ===")
class SecurityFSMv2:
    def __init__(self):
        self.state="PATROL"; self.battery=100.; self.alert_t=0; self.patrol_pos=0
    def step(self, percept):
        self.battery -= 2.
        if self.state != "RECHARGE" and self.battery <= 20:
            self.state = "RECHARGE"
        if self.state == "RECHARGE":
            self.battery = min(100., self.battery+5.)
            if self.battery >= 100.: self.state = "PATROL"
            return self.state, f"charging ({self.battery:.0f}%)"
        if self.state == "PATROL":
            self.patrol_pos = (self.patrol_pos+1)%10
            if percept.get("threat"): self.state="ALERT"; self.alert_t=0
        elif self.state == "ALERT":
            self.alert_t+=1
            if percept.get("confirmed"): self.state="PATROL"
            elif self.alert_t>3: self.state="PATROL"
        return self.state, f"battery={self.battery:.0f}"

random.seed(2); bot = SecurityFSMv2()
print(f"{'Step':>4}  {'State':>10}  Detail")
print("-"*38)
for i in range(20):
    p={"threat":random.random()<0.3,"confirmed":random.random()<0.5}
    s,d = bot.step(p)
    print(f"{i:>4}  {s:>10}  {d}")

# Exercise 3 — Utility-based agent
print("\n=== Exercise 3: Utility Agent ===")
objects = {"A":(10,8),"B":(7,3),"C":(3,12)}  # name: (utility, pos)
robot_pos = 0; remaining = dict(objects); total_net = 0; order=[]
while remaining:
    best_name, best_net = None, -float("inf")
    for name,(util,pos) in remaining.items():
        net = util - abs(pos - robot_pos)
        if net > best_net: best_net=net; best_name=name
    util,pos = remaining.pop(best_name)
    total_net += util - abs(pos-robot_pos)
    print(f"  Collect {best_name} at pos={pos}, utility={util}, travel={abs(pos-robot_pos)}, net={util-abs(pos-robot_pos)}")
    robot_pos=pos; order.append(best_name)
print(f"  Order: {order}, total net utility = {total_net}")

# Exercise 4 — Cooperative vacuum
print("\n=== Exercise 4: Cooperative Vacuum ===")
grid=[["D"]*4 for _ in range(4)]
pos1,pos2=[0,0],[3,3]; steps1=steps2=0
MOVES=[(-1,0),(1,0),(0,-1),(0,1)]
import random as R; R.seed(3)
def nearest_dirty_excl(grid,pos,excl):
    best,bd=None,float("inf")
    for r in range(4):
        for c in range(4):
            if grid[r][c]=="D" and [r,c]!=excl:
                d=abs(r-pos[0])+abs(c-pos[1])
                if d<bd: bd=d;best=[r,c]
    return best
for _ in range(100):
    for agent,(pos,other) in enumerate([(pos1,pos2),(pos2,pos1)]):
        r,c=pos
        if grid[r][c]=="D": grid[r][c]="C"; (steps1 if agent==0 else steps2).__class__  # count below
        target=nearest_dirty_excl(grid,pos,other)
        if target is None: continue
        dr=target[0]-r; dc=target[1]-c
        if abs(dr)>=abs(dc): nr,nc=r+(1 if dr>0 else -1),c
        else: nr,nc=r,c+(1 if dc>0 else -1)
        if [nr,nc]!=other: pos[:]=nr,nc
    steps1+=1; steps2+=1
    if all(grid[r][c]=="C" for r in range(4) for c in range(4)): break
print(f"  All clean in ~{steps1} global steps")

# Exercise 5 — Belief state agent
print("\n=== Exercise 5: Belief State Agent ===")
true_state = {0:"open",1:"closed",2:"open",3:"closed",4:"open"}
belief = {i:"unknown" for i in range(5)}
pos = 0
print(f"  Initial belief: {belief}")
for step in range(10):
    belief[pos] = true_state[pos]
    if all(v!="unknown" for v in belief.values()): break
    # move to nearest unknown
    unknowns = [d for d,v in belief.items() if v=="unknown"]
    if not unknowns: break
    pos = min(unknowns, key=lambda d: abs(d-pos))
    print(f"  Step {step+1}: observed door {pos}={true_state[pos]}, belief={belief}")
print(f"  Final belief: {belief}")
