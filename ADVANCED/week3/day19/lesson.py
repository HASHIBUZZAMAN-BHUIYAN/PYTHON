# Advanced Day 19 — Intro to Agentic AI: Rule-Based Agents
# ~10 MB RAM, <1s on CPU

import random
import time

print("""
=== Agentic AI — Day 19 ===

An AGENT is a system that:
  1. Perceives its environment (observations / state)
  2. Decides on an action (using rules, planning, or learning)
  3. Acts on the environment
  4. Loops — repeating until a goal is reached or terminated

Types of agents (rule-based):
  • Simple reflex agent   — if <condition> then <action>
  • Model-based reflex    — maintains internal state + rules
  • Goal-based agent      — plans to reach a goal state
  • Utility-based agent   — maximizes a utility function
  • Learning agent        — improves its rules over time (→ Day 17 Q-learning)
""")

# ─── 1. SIMPLE REFLEX AGENT ──────────────────────────────────────────────────
print("=== 1. Simple Reflex Agent — Thermostat ===\n")

class ThermostatAgent:
    """
    Percept: current temperature.
    Rules:
      if temp < setpoint-1 → HEAT ON
      if temp > setpoint+1 → COOL ON
      else                 → IDLE
    """
    def __init__(self, setpoint=22.0):
        self.setpoint = setpoint

    def perceive_decide(self, temperature):
        if temperature < self.setpoint - 1:
            return "HEAT_ON"
        elif temperature > self.setpoint + 1:
            return "COOL_ON"
        return "IDLE"

def run_thermostat():
    agent = ThermostatAgent(setpoint=22.0)
    temp = 16.0
    print(f"{'Step':>4}  {'Temp':>6}  Action")
    print("-" * 26)
    for step in range(12):
        action = agent.perceive_decide(temp)
        print(f"{step:>4}  {temp:>6.2f}  {action}")
        if action == "HEAT_ON":   temp += 1.5
        elif action == "COOL_ON": temp -= 1.2
        else:                     temp += random.uniform(-0.3, 0.3)

run_thermostat()

# ─── 2. FINITE STATE MACHINE AGENT ───────────────────────────────────────────
print("\n=== 2. FSM Agent — Security Robot ===\n")
print("""
States: PATROL → ALERT → PURSUE → RETURN
  PATROL: move along fixed path; if threat detected → ALERT
  ALERT:  confirm threat; if confirmed → PURSUE; else → PATROL
  PURSUE: chase threat; if caught or lost → RETURN
  RETURN: go back to base; when arrived → PATROL
""")

class SecurityRobotFSM:
    STATES = ["PATROL","ALERT","PURSUE","RETURN"]

    def __init__(self):
        self.state      = "PATROL"
        self.patrol_pos = 0
        self.alert_ticks= 0
        self.pursuit_ticks = 0

    def step(self, percept):
        """percept: dict with 'threat_detected', 'threat_confirmed', 'threat_caught', 'at_base'"""
        prev = self.state

        if self.state == "PATROL":
            self.patrol_pos = (self.patrol_pos + 1) % 10
            if percept.get("threat_detected"):
                self.state = "ALERT"; self.alert_ticks = 0

        elif self.state == "ALERT":
            self.alert_ticks += 1
            if percept.get("threat_confirmed"):
                self.state = "PURSUE"; self.pursuit_ticks = 0
            elif self.alert_ticks >= 3:
                self.state = "PATROL"   # false alarm

        elif self.state == "PURSUE":
            self.pursuit_ticks += 1
            if percept.get("threat_caught") or self.pursuit_ticks > 8:
                self.state = "RETURN"

        elif self.state == "RETURN":
            if percept.get("at_base"):
                self.state = "PATROL"

        action = {
            "PATROL": f"moving to waypoint {self.patrol_pos}",
            "ALERT":  f"scanning (tick {self.alert_ticks})",
            "PURSUE": f"chasing threat (tick {self.pursuit_ticks})",
            "RETURN": "returning to base",
        }[self.state]
        return prev, self.state, action

robot = SecurityRobotFSM()
scenario = [
    {"threat_detected":False,"threat_confirmed":False,"threat_caught":False,"at_base":False},
    {"threat_detected":False,"threat_confirmed":False,"threat_caught":False,"at_base":False},
    {"threat_detected":True, "threat_confirmed":False,"threat_caught":False,"at_base":False},
    {"threat_detected":True, "threat_confirmed":True, "threat_caught":False,"at_base":False},
    {"threat_detected":True, "threat_confirmed":True, "threat_caught":False,"at_base":False},
    {"threat_detected":True, "threat_confirmed":True, "threat_caught":True, "at_base":False},
    {"threat_detected":False,"threat_confirmed":False,"threat_caught":False,"at_base":False},
    {"threat_detected":False,"threat_confirmed":False,"threat_caught":False,"at_base":True},
    {"threat_detected":False,"threat_confirmed":False,"threat_caught":False,"at_base":False},
]
print(f"{'Step':>4}  {'From':>8} → {'To':>8}  Action")
print("-" * 55)
for i, percept in enumerate(scenario):
    prev, new, action = robot.step(percept)
    transition = "  " if prev==new else "→"
    print(f"{i:>4}  {prev:>8} {transition} {new:>8}  {action}")

# ─── 3. GOAL-BASED AGENT ─────────────────────────────────────────────────────
print("\n=== 3. Goal-Based Agent — Grid World ===\n")
print("""
Agent navigates a 5×5 grid from S to G using a greedy policy (Manhattan distance).
State = (row, col). No obstacles.
""")

class GoalAgent:
    MOVES = [(-1,0),(1,0),(0,-1),(0,1)]
    NAMES = ["UP","DOWN","LEFT","RIGHT"]

    def __init__(self, start, goal, grid_size=5):
        self.pos  = list(start)
        self.goal = goal
        self.n    = grid_size
        self.memory = [tuple(start)]

    def step(self):
        if tuple(self.pos) == self.goal:
            return None, "DONE"
        # pick action minimizing Manhattan distance
        best_a, best_dist, best_pos = None, float("inf"), None
        for i,(dr,dc) in enumerate(self.MOVES):
            nr,nc = self.pos[0]+dr, self.pos[1]+dc
            if 0<=nr<self.n and 0<=nc<self.n:
                d = abs(nr-self.goal[0]) + abs(nc-self.goal[1])
                if d < best_dist:
                    best_dist=d; best_a=i; best_pos=[nr,nc]
        self.pos = best_pos
        self.memory.append(tuple(self.pos))
        return self.NAMES[best_a], f"moved to {tuple(self.pos)}"

agent = GoalAgent((0,0),(4,4))
print(f"{'Step':>4}  Action    State")
print("-"*28)
for i in range(15):
    action, msg = agent.step()
    print(f"{i:>4}  {(action or 'STOP'):>8}  {msg}")
    if action is None: break

# ─── 4. MODEL-BASED REFLEX AGENT ─────────────────────────────────────────────
print("\n=== 4. Model-Based Reflex Agent — Robot Vacuum ===\n")
print("""
Environment: 3×3 grid. Each cell is 'clean' or 'dirty'.
Agent maintains an internal map, sucks when on a dirty cell,
moves toward the nearest dirty cell otherwise.
""")

class VacuumAgent:
    def __init__(self, grid):
        self.grid = [row[:] for row in grid]  # internal model (copy)
        self.pos  = [0, 0]
        self.steps = 0

    def nearest_dirty(self):
        best, best_d = None, float("inf")
        for r in range(len(self.grid)):
            for c in range(len(self.grid[0])):
                if self.grid[r][c] == "D":
                    d = abs(r-self.pos[0]) + abs(c-self.pos[1])
                    if d < best_d: best_d=d; best=[r,c]
        return best

    def step(self):
        self.steps += 1
        r,c = self.pos
        if self.grid[r][c] == "D":
            self.grid[r][c] = "C"
            return "SUCK", f"cleaned ({r},{c})"
        target = self.nearest_dirty()
        if target is None:
            return "STOP","all clean"
        dr = target[0]-r; dc = target[1]-c
        if abs(dr) >= abs(dc):
            self.pos[0] += (1 if dr>0 else -1)
        else:
            self.pos[1] += (1 if dc>0 else -1)
        return "MOVE", f"moved to {tuple(self.pos)}"

initial_grid = [
    ["D","C","D"],
    ["C","D","C"],
    ["D","C","D"],
]
vacuum = VacuumAgent(initial_grid)
print(f"{'Step':>4}  Action  Detail")
print("-"*35)
for _ in range(20):
    action, detail = vacuum.step()
    print(f"{vacuum.steps:>4}  {action:>6}  {detail}")
    if action == "STOP": break
