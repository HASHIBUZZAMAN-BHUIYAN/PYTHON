"""
Project: Autonomous Delivery Agent
Teaches: multi-stop routing (greedy nearest neighbor + 2-opt improvement),
         delivery simulation with constraints, route visualization.
~20 MB RAM, ~1s on CPU
"""
import math, random
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Package:
    pkg_id:   str
    pickup:   tuple
    dropoff:  tuple
    priority: int   # 1=urgent, 2=normal, 3=low
    weight:   float

class DeliveryAgent:
    MAX_LOAD = 20.0
    def __init__(self, start=(5,5)):
        self.pos    = start; self.start = start
        self.route  = []; self.history = [start]
        self.load   = 0.0; self.delivered = 0; self.steps = 0

    @staticmethod
    def dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

    def plan_route(self, packages: List[Package]):
        # Greedy nearest-neighbor (by priority then distance)
        pending = sorted(packages, key=lambda p: (p.priority, self.dist(self.pos, p.pickup)))
        self.route = []
        for pkg in pending:
            self.route.append(("PICKUP",  pkg.pkg_id, pkg.pickup,  pkg.weight))
            self.route.append(("DROPOFF", pkg.pkg_id, pkg.dropoff, -pkg.weight))
        return self.route

    def two_opt(self, positions: List[tuple]) -> List[tuple]:
        """Simple 2-opt improvement on position sequence."""
        if len(positions) < 4: return positions
        improved = True; route = positions[:]
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if (self.dist(route[i-1],route[i])+self.dist(route[j-1],route[j])) > \
                       (self.dist(new_route[i-1],new_route[i])+self.dist(new_route[j-1],new_route[j])):
                        route = new_route; improved = True
        return route

    def execute(self, packages: List[Package]):
        self.plan_route(packages)
        total_dist = 0.0
        pkg_map    = {p.pkg_id: p for p in packages}
        cargo: dict= {}

        for action, pkg_id, location, delta in self.route:
            dist_to = self.dist(self.pos, location)
            total_dist += dist_to
            self.history.append(location)
            self.pos = location; self.steps += 1

            if action == "PICKUP":
                if self.load + pkg_map[pkg_id].weight <= self.MAX_LOAD:
                    cargo[pkg_id] = pkg_map[pkg_id]; self.load += delta
                    print(f"  PICKUP  {pkg_id} at {location}  load={self.load:.1f}  dist+={dist_to:.1f}")
                else:
                    print(f"  SKIP    {pkg_id} — overload ({self.load:.1f}+{delta:.1f}>{self.MAX_LOAD})")
            else:
                if pkg_id in cargo:
                    self.load += delta; cargo.pop(pkg_id); self.delivered += 1
                    print(f"  DROPOFF {pkg_id} at {location}  load={self.load:.1f}  dist+={dist_to:.1f}")

        # Return to depot
        dist_home = self.dist(self.pos, self.start)
        total_dist += dist_home; self.history.append(self.start); self.pos = self.start
        print(f"\n  RETURN  to depot at {self.start}  dist+={dist_home:.1f}")
        return total_dist

random.seed(7)
PACKAGES = [
    Package("P1", (1,1), (8,8), 1, 3.0),
    Package("P2", (2,8), (6,2), 2, 5.0),
    Package("P3", (7,1), (3,7), 2, 4.0),
    Package("P4", (4,4), (9,9), 1, 8.0),
    Package("P5", (1,6), (5,5), 3, 2.0),
    Package("P6", (8,3), (2,2), 3, 6.0),
]
print("=== Delivery Agent Simulation ===\n")
agent = DeliveryAgent(start=(5,5))
total_dist = agent.execute(PACKAGES)
print(f"\n  Delivered: {agent.delivered}/{len(PACKAGES)}  Total distance: {total_dist:.1f}  Steps: {agent.steps}")

# ─── Visualization ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = plt.cm.tab10.colors

ax = axes[0]
ax.set_xlim(-1, 11); ax.set_ylim(-1, 11)
for i, pkg in enumerate(PACKAGES):
    c = colors[i % 10]
    ax.plot(*pkg.pickup[::-1],  "^", color=c, markersize=10)
    ax.plot(*pkg.dropoff[::-1], "v", color=c, markersize=10)
    ax.annotate(pkg.pkg_id, pkg.pickup[::-1],  textcoords="offset points", xytext=(4,4),  fontsize=7)
    ax.annotate(pkg.pkg_id, pkg.dropoff[::-1], textcoords="offset points", xytext=(4,-8), fontsize=7)
hist = agent.history
xs=[p[1] for p in hist]; ys=[p[0] for p in hist]
ax.plot(xs, ys, "k--", alpha=0.4, linewidth=1)
ax.plot(xs[0], ys[0], "gs", markersize=14, label="Depot")
ax.grid(True, alpha=0.3); ax.legend(fontsize=8); ax.set_title("Delivery Route", fontsize=10)

# Route summary bar
ax2 = axes[1]
pkg_labels = [p.pkg_id for p in PACKAGES]
priorities = [p.priority for p in PACKAGES]
weights    = [p.weight   for p in PACKAGES]
x = range(len(PACKAGES))
bars = ax2.bar(pkg_labels, weights, color=[{1:"tomato",2:"steelblue",3:"gray"}[p] for p in priorities])
ax2.set_title("Package Weights by Priority\n(red=urgent, blue=normal, gray=low)"); ax2.set_ylabel("Weight (kg)")
for bar, w in zip(bars, weights): ax2.text(bar.get_x()+bar.get_width()/2, w+0.1, f"{w}kg", ha="center", fontsize=8)
ax2.grid(axis="y", alpha=0.3)

plt.suptitle(f"Delivery Agent  |  {agent.delivered}/{len(PACKAGES)} delivered  |  Total dist={total_dist:.1f}", fontsize=11)
plt.tight_layout(); plt.savefig("delivery_agent.png", dpi=90); plt.close()
print("Saved delivery_agent.png")
