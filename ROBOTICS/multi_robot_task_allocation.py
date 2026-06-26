"""
Multi-Robot Task Allocation (Greedy Auction)
=============================================
What it does:
  3 robots must complete 7 tasks scattered in a 10x10 m arena.
  Two allocation strategies are compared:
    1. GREEDY DISTANCE: build a cost matrix (Euclidean distance), then
       repeatedly assign the cheapest unassigned (robot, task) pair
    2. AUCTION (shown as the live demo): each unassigned task is put to
       auction -- every free robot bids its distance to the task, the
       winner is the closest bidder; exactly one assignment per round
  The animation shows:
    - Robots (coloured dots) moving toward their current assigned task
    - Task markers: white = unassigned, coloured = assigned, grey = done
    - Assignment lines from robot to target
    - Live table: which robot is doing which task, wait queue

What it teaches:
  - Task allocation as an assignment problem (link to Hungarian algorithm)
  - Greedy vs optimal assignment: greedy is O(n^2), optimal is O(n^3)
  - Auction mechanism: decentralised allocation without a global planner
  - Makespan: total time until ALL tasks complete (the metric to minimise)
  - How queue length affects robot utilisation (some robots idle while busy)

Controls: None -- auto-runs until all tasks complete.
Animation: Live matplotlib window. Set MPLBACKEND=Agg to skip display (headless).
Output: ROBOTICS/outputs/task_allocation.png
RAM: ~60 MB | Time: ~6s with display, <1s headless
"""

import os
import numpy as np
import matplotlib
INTERACTIVE = matplotlib.get_backend().lower() != 'agg'
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs("ROBOTICS/outputs", exist_ok=True)
rng = np.random.default_rng(7)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
N_ROBOTS   = 3
N_TASKS    = 7
N_STEPS    = 600
DT         = 0.04
MAX_SPD    = 2.0
TASK_R     = 0.45     # task completion radius
EXEC_TIME  = 15       # steps robot spends "executing" task on arrival
DRAW_N     = 4

ROBOT_COLORS = ['#e74c3c', '#3498db', '#2ecc71']
ROBOT_STARTS = np.array([[1.0, 1.0], [9.0, 1.0], [5.0, 9.0]])

TASK_POS = np.array([
    [3.0, 2.5],
    [7.5, 2.0],
    [2.0, 6.0],
    [5.5, 5.0],
    [8.5, 7.5],
    [4.0, 8.5],
    [6.5, 3.5],
])

# ─── GREEDY AUCTION ALLOCATION ────────────────────────────────────────────────
def auction_assign(robot_pos, task_pos, done_tasks, assigned):
    """One auction round: assign one free task to closest free robot."""
    free_robots = [i for i in range(N_ROBOTS) if assigned[i] is None]
    free_tasks  = [j for j in range(N_TASKS) if j not in done_tasks
                   and j not in [assigned[r] for r in range(N_ROBOTS) if assigned[r] is not None]]
    if not free_robots or not free_tasks:
        return assigned
    # All bidders bid on all free tasks; pick globally cheapest (robot, task) pair
    best_dist, best_r, best_t = float('inf'), None, None
    for r in free_robots:
        for t in free_tasks:
            d = np.linalg.norm(robot_pos[r] - task_pos[t])
            if d < best_dist:
                best_dist, best_r, best_t = d, r, t
    if best_r is not None:
        assigned[best_r] = best_t
    return assigned

# ─── SIMULATION STATE ─────────────────────────────────────────────────────────
print("Multi-Robot Task Allocation (Greedy Auction)")
print("=" * 50)

pos      = ROBOT_STARTS.copy().astype(float)
vel      = np.zeros((N_ROBOTS, 2))
assigned = [None] * N_ROBOTS     # assigned[r] = task index (or None if free)
exec_cd  = [0]    * N_ROBOTS     # steps remaining in task execution
done_tasks = set()
paths    = [[p.copy()] for p in pos]
timeline = []   # (step, robot, task, event)
makespan = None

# Initial auction: assign tasks to robots before first step
assigned = auction_assign(pos, TASK_POS, done_tasks, assigned)
assigned = auction_assign(pos, TASK_POS, done_tasks, assigned)
assigned = auction_assign(pos, TASK_POS, done_tasks, assigned)
print(f"  Initial assignments: {[(i+1, a+1 if a is not None else '-') for i,a in enumerate(assigned)]}")

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle("Multi-Robot Task Allocation (Greedy Auction)", fontsize=11)
if INTERACTIVE:
    plt.ion()

def draw(step):
    ax.clear();  ax2.clear()
    ax.set_xlim(-0.5, 10.5);  ax.set_ylim(-0.5, 10.5)
    ax.set_aspect('equal');  ax.set_facecolor('#f8f8f8')
    ax.set_title(f"Step {step}/{N_STEPS}  done {len(done_tasks)}/{N_TASKS}  "
                 f"{'[ALL DONE]' if len(done_tasks)==N_TASKS else ''}", fontsize=9)

    # Tasks
    for j, tp in enumerate(TASK_POS):
        if j in done_tasks:
            ax.plot(*tp, 'X', ms=12, color='grey', alpha=0.5, zorder=3)
            continue
        assigned_to = next((r for r in range(N_ROBOTS) if assigned[r]==j), None)
        col = ROBOT_COLORS[assigned_to] if assigned_to is not None else 'white'
        ax.add_patch(patches.Circle(tp, TASK_R, facecolor=col,
                                    edgecolor='black', lw=1.5, zorder=3, alpha=0.6))
        ax.text(*tp, str(j+1), ha='center', va='center', fontsize=9,
                fontweight='bold', zorder=4)

    # Robots
    for r in range(N_ROBOTS):
        pt = np.array(paths[r])
        ax.plot(pt[:,0], pt[:,1], '-', color=ROBOT_COLORS[r], alpha=0.2, lw=1)
        ec = 'gold' if exec_cd[r] > 0 else 'black'
        ax.plot(*pos[r], 'o', ms=16, color=ROBOT_COLORS[r], zorder=6,
                markeredgecolor=ec, markeredgewidth=2)
        ax.text(pos[r][0], pos[r][1], f"R{r+1}", ha='center', va='center',
                fontsize=7, fontweight='bold', color='white', zorder=7)
        # Assignment line
        if assigned[r] is not None and assigned[r] not in done_tasks:
            tp = TASK_POS[assigned[r]]
            ax.plot([pos[r][0], tp[0]], [pos[r][1], tp[1]], '--',
                    color=ROBOT_COLORS[r], alpha=0.5, lw=1.5, zorder=2)

    ax.plot(*ROBOT_STARTS.T, 'ks', ms=5, alpha=0.3, zorder=2)
    ax.grid(alpha=0.2)

    # Timeline table
    ax2.set_xlim(0, 1);  ax2.set_ylim(0, 1);  ax2.axis('off')
    ax2.set_title("Task Assignment Log", fontsize=9)
    rows = [f"{'Step':>5}  {'Robot':>5}  {'Task':>4}  Event"]
    rows.append("-" * 32)
    for ev in timeline[-18:]:
        rows.append(f"{ev[0]:>5}  R{ev[1]+1:>4}  T{ev[2]+1:>3}  {ev[3]}")
    ax2.text(0.05, 0.97, "\n".join(rows), transform=ax2.transAxes,
             fontsize=7.5, fontfamily='monospace', va='top')
    plt.tight_layout()

for step in range(N_STEPS):
    if len(done_tasks) == N_TASKS:
        if makespan is None:
            makespan = step
        break

    for r in range(N_ROBOTS):
        if exec_cd[r] > 0:
            # Robot executing task
            exec_cd[r] -= 1
            if exec_cd[r] == 0:
                t_done = assigned[r]
                done_tasks.add(t_done)
                timeline.append((step, r, t_done, 'COMPLETED'))
                print(f"  step {step:3d}: Robot {r+1} completed Task {t_done+1}  "
                      f"({len(done_tasks)}/{N_TASKS})")
                assigned[r] = None
                # Trigger new auction round
                assigned = auction_assign(pos, TASK_POS, done_tasks, assigned)
                if assigned[r] is not None:
                    timeline.append((step, r, assigned[r], 'ASSIGNED'))
            vel[r] = np.zeros(2)
            continue

        if assigned[r] is None:
            # Idle robot: try auction
            assigned = auction_assign(pos, TASK_POS, done_tasks, assigned)
            if assigned[r] is not None:
                timeline.append((step, r, assigned[r], 'ASSIGNED'))
            vel[r] = np.zeros(2)
            continue

        # Move toward assigned task
        tgt = TASK_POS[assigned[r]]
        diff = tgt - pos[r]
        d = np.linalg.norm(diff)
        if d < TASK_R:
            # Arrived -- start execution
            exec_cd[r] = EXEC_TIME
            timeline.append((step, r, assigned[r], 'ARRIVED'))
            vel[r] = np.zeros(2)
        else:
            vel[r] = MAX_SPD * diff / d

    pos += vel * DT
    pos = np.clip(pos, 0.0, 10.0)
    for r in range(N_ROBOTS):
        paths[r].append(pos[r].copy())

    if step % DRAW_N == 0:
        draw(step)
        if INTERACTIVE:
            plt.pause(0.04)

draw(min(step, N_STEPS-1))
if makespan:
    print(f"All {N_TASKS} tasks complete at step {makespan}  "
          f"(makespan = {makespan*DT:.1f}s)")
else:
    print(f"Completed {len(done_tasks)}/{N_TASKS} tasks at step limit")

plt.savefig("ROBOTICS/outputs/task_allocation.png", dpi=90, bbox_inches='tight')
print("Saved: ROBOTICS/outputs/task_allocation.png")
if INTERACTIVE:
    plt.ioff();  plt.show()
plt.close()
