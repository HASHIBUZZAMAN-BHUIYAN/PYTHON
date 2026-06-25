# Advanced Day 19 Mini-Project — Autonomous Warehouse Robot Agent
# Rule-based FSM agent manages a warehouse: pick orders, charge, avoid collisions.
# ~15 MB RAM, <2s on CPU

import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

print("=== Day 19 Mini-Project: Warehouse Robot Agent ===\n")

# ─── Warehouse layout ────────────────────────────────────────────────────────
# 8×8 grid
# Shelf cells (S), Charging station (C), Dispatch area (D), Open floor (.)
LAYOUT = [
    [".",".",".",".",".",".",".","."],
    [".","S","S",".","S","S",".","."],
    [".","S","S",".","S","S",".","."],
    [".",".",".",".",".",".",".","."],
    [".","S","S",".","S","S",".","."],
    [".","S","S",".","S","S",".","."],
    [".",".",".",".",".",".",".","."],
    ["C",".",".",".",".",".",".","D"],
]
ROWS=COLS=8

SHELF_CELLS  = [(r,c) for r in range(ROWS) for c in range(COLS) if LAYOUT[r][c]=="S"]
DISPATCH     = (7,7)
CHARGER      = (7,0)

# ─── Order queue ─────────────────────────────────────────────────────────────
random.seed(42)
ORDER_QUEUE = [random.choice(SHELF_CELLS) for _ in range(12)]
print(f"Order queue ({len(ORDER_QUEUE)} orders): {ORDER_QUEUE}\n")

# ─── Manhattan move helper ────────────────────────────────────────────────────
def move_toward(pos, target):
    r,c = pos; tr,tc = target
    dr,dc = tr-r, tc-c
    if abs(dr) >= abs(dc): return (r+(1 if dr>0 else -1), c)
    return (r, c+(1 if dc>0 else -1))

# ─── Warehouse Robot FSM ──────────────────────────────────────────────────────
class WarehouseRobot:
    STATES = ["IDLE","NAVIGATE_SHELF","PICK","NAVIGATE_DISPATCH","DEPOSIT","NAVIGATE_CHARGE","CHARGING"]

    def __init__(self, name, start_pos, battery=100.):
        self.name        = name
        self.pos         = start_pos
        self.battery     = battery
        self.state       = "IDLE"
        self.target_shelf= None
        self.carrying    = False
        self.orders_done = 0
        self.history     = [pos]
        self.state_history = ["IDLE"]

    def assign_order(self, shelf):
        if self.state == "IDLE":
            self.target_shelf = shelf
            self.state = "NAVIGATE_SHELF"

    def step(self, order_queue):
        drain = {"IDLE":0.5,"NAVIGATE_SHELF":2.,"PICK":1.,"NAVIGATE_DISPATCH":2.,
                 "DEPOSIT":1.,"NAVIGATE_CHARGE":2.,"CHARGING":-8.}
        self.battery = min(100., max(0., self.battery - drain[self.state]))

        # Emergency charge check
        if self.battery <= 15. and self.state not in ("NAVIGATE_CHARGE","CHARGING"):
            self.carrying = False  # drop item
            self.state = "NAVIGATE_CHARGE"

        action = "wait"
        if self.state == "IDLE":
            if order_queue:
                self.assign_order(order_queue.pop(0))
            action = "waiting for order"

        elif self.state == "NAVIGATE_SHELF":
            if self.pos == self.target_shelf:
                self.state = "PICK"
            else:
                self.pos = move_toward(self.pos, self.target_shelf)
            action = f"moving to shelf {self.target_shelf}"

        elif self.state == "PICK":
            self.carrying = True
            self.state = "NAVIGATE_DISPATCH"
            action = f"picked item at {self.pos}"

        elif self.state == "NAVIGATE_DISPATCH":
            if self.pos == DISPATCH:
                self.state = "DEPOSIT"
            else:
                self.pos = move_toward(self.pos, DISPATCH)
            action = "moving to dispatch"

        elif self.state == "DEPOSIT":
            self.carrying = False
            self.orders_done += 1
            self.state = "IDLE" if self.battery > 25. else "NAVIGATE_CHARGE"
            action = f"deposited (total={self.orders_done})"

        elif self.state == "NAVIGATE_CHARGE":
            if self.pos == CHARGER:
                self.state = "CHARGING"
            else:
                self.pos = move_toward(self.pos, CHARGER)
            action = "navigating to charger"

        elif self.state == "CHARGING":
            if self.battery >= 90.:
                self.state = "IDLE"
            action = f"charging ({self.battery:.0f}%)"

        self.history.append(self.pos)
        self.state_history.append(self.state)
        return action

# ─── Simulate two robots ──────────────────────────────────────────────────────
robot_A = WarehouseRobot("A", (0,0), battery=100.)
robot_B = WarehouseRobot("B", (0,7), battery=70.)

queue = ORDER_QUEUE.copy()

print(f"{'Step':>4}  {'Bot':>3}  {'Battery':>7}  {'State':>20}  {'Pos':>8}  Action")
print("-"*72)

metrics = {"A_orders":[],"B_orders":[]}
for step in range(80):
    for robot in [robot_A, robot_B]:
        action = robot.step(queue)
        if step % 5 == 0 or robot.state in ("PICK","DEPOSIT","CHARGING"):
            print(f"{step:>4}  {robot.name:>3}  {robot.battery:>7.1f}"
                  f"  {robot.state:>20}  {str(robot.pos):>8}  {action}")
    metrics["A_orders"].append(robot_A.orders_done)
    metrics["B_orders"].append(robot_B.orders_done)
    if robot_A.state=="IDLE" and robot_B.state=="IDLE" and not queue:
        print(f"\n  All {len(ORDER_QUEUE)} orders fulfilled at step {step}!")
        break

print(f"\n  Robot A: {robot_A.orders_done} orders, final battery={robot_A.battery:.1f}%")
print(f"  Robot B: {robot_B.orders_done} orders, final battery={robot_B.battery:.1f}%")

# ─── Visualize ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Warehouse heatmap of visitation
def draw_warehouse(ax, robot, title):
    grid = [[0]*COLS for _ in range(ROWS)]
    for (r,c) in robot.history: grid[r][c] += 1
    import numpy as np
    arr = np.array(grid, dtype=float)
    ax.imshow(arr, cmap="YlOrRd", interpolation="nearest", vmin=0)
    # overlay layout
    for r in range(ROWS):
        for c in range(COLS):
            cell = LAYOUT[r][c]
            if cell == "S": ax.text(c,r,"S",ha="center",va="center",fontsize=8,color="navy")
            elif cell == "C": ax.text(c,r,"C",ha="center",va="center",fontsize=9,color="green",fontweight="bold")
            elif cell == "D": ax.text(c,r,"D",ha="center",va="center",fontsize=9,color="purple",fontweight="bold")
    ax.set_title(title); ax.axis("off")

draw_warehouse(axes[0], robot_A, f"Robot A — {robot_A.orders_done} orders")
draw_warehouse(axes[1], robot_B, f"Robot B — {robot_B.orders_done} orders")

plt.suptitle("Day 19 Mini-Project — Warehouse Robot Agent", fontsize=12)
plt.tight_layout(); plt.savefig("warehouse_agent.png", dpi=90)
print("\nSaved warehouse_agent.png")
plt.show()
