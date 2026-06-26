"""
ROS-Style Publish/Subscribe Architecture (Pure Python)
=======================================================
REAL TOOL THIS MODELS:
  ROS (Robot Operating System) and ROS 2 -- the de-facto standard middleware
  for writing robot software.  A ROS system is a graph of "nodes" (processes)
  that exchange typed "messages" over named "topics" using a publish/subscribe
  model.  ROS 2 uses DDS (Data Distribution Service) as its transport layer,
  giving it automatic node discovery, quality-of-service settings, and
  cross-machine/cross-language communication.

WHAT THIS SCRIPT DOES (the honest, working version):
  Implements the core pub-sub pattern in pure Python -- no ROS installation
  needed -- with a synchronous single-process message bus:
    - MessageBus: a broker that routes published messages to subscribers
    - Node base class: create_publisher() and create_subscription() methods
    - Four derived nodes mimicking a real ROS robot system:
        OdometryNode  -- publishes /odom (simulated robot pose)
        LidarNode     -- publishes /scan (8 simulated range readings)
        PlannerNode   -- subscribes /odom + /scan; publishes /cmd_vel
        MotorNode     -- subscribes /cmd_vel; updates simulated robot state
  The planner uses a simple proportional heading controller to steer the
  robot from start to goal.  A live animation shows the robot moving in
  the 2D world while a message-traffic panel logs each published message.

WHAT REAL ROS/ROS2 ADDS (beyond this script):
  - Full typed message system (msg files compiled to language-specific structs)
  - ROS master (ROS 1) or DDS peer discovery (ROS 2) for cross-process comms
  - Launch files (.launch.py) for starting/configuring multi-node systems
  - Parameter server, TF (transform tree), service calls, actions (not topics)
  - Cross-language support (C++, Python, Rust, Java)
  - Cross-machine networking (robots, PCs, cloud)
  - Visualization: RViz, rqt_graph (shows the pub-sub graph live)
  To learn real ROS: ros.org (free) -- runs on Ubuntu/WSL2 with standard hardware.

THIS SCRIPT RUNS ON: any Python 3.9+ environment with numpy + matplotlib.
No ROS installation, no GPU, no special hardware needed. CPU-only, ~60 MB RAM.

Output: ROBOTICS/ADVANCED_NVIDIA_ROS/outputs/ros_pubsub.png
"""

import os, collections, math
import numpy as np
import matplotlib
INTERACTIVE = matplotlib.get_backend().lower() != 'agg'
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs("ROBOTICS/ADVANCED_NVIDIA_ROS/outputs", exist_ok=True)
rng = np.random.default_rng(7)

# ─── MESSAGE BUS (the "ROS master / DDS" equivalent) ─────────────────────────
class MessageBus:
    """Central broker routing published messages to all subscribers of a topic."""
    def __init__(self):
        self._subs = collections.defaultdict(list)   # topic -> [callbacks]
        self._msg_log = []                            # (time, sender, topic, preview)
        self._time = 0.0

    def publish(self, topic, msg, sender_name="?"):
        preview = str(msg)[:55]
        self._msg_log.append((self._time, sender_name, topic, preview))
        for cb in self._subs[topic]:
            cb(msg)

    def subscribe(self, topic, callback):
        self._subs[topic].append(callback)

    def tick(self, dt):
        self._time += dt

    @property
    def time(self):
        return self._time

# Global bus (in a real ROS system this is the DDS domain)
BUS = MessageBus()

# ─── NODE BASE CLASS ──────────────────────────────────────────────────────────
class Node:
    def __init__(self, name):
        self.name = name
        self._pub_counts = collections.defaultdict(int)

    def create_publisher(self, topic):
        def publish(msg):
            self._pub_counts[topic] += 1
            BUS.publish(topic, msg, self.name)
        return publish

    def create_subscription(self, topic, callback):
        BUS.subscribe(topic, callback)

    def step(self):  # override in subclasses
        pass

# ─── SIMULATED ROBOT WORLD ────────────────────────────────────────────────────
ARENA  = 10.0
GOAL   = np.array([9.0, 9.0])
# Obstacles placed beside the diagonal so the heading controller can reach the goal
OBSTACLES = [(1.5,5.0,0.5), (5.5,2.0,0.5), (8.5,5.0,0.5)]

def scan_from_pose(x, y, theta, n_beams=8, max_r=5.0):
    """Simulated lidar: 8 rays, return distances to nearest obstacle/wall."""
    readings = []
    for i in range(n_beams):
        angle = theta + i * (2*math.pi / n_beams)
        dx, dy = math.cos(angle), math.sin(angle)
        dist = max_r
        for r_step in np.arange(0.1, max_r, 0.1):
            sx = x + dx*r_step; sy = y + dy*r_step
            if not (0 <= sx <= ARENA and 0 <= sy <= ARENA):
                dist = r_step; break
            for ox, oy, or_ in OBSTACLES:
                if math.hypot(sx-ox, sy-oy) < or_:
                    dist = r_step; break
            else:
                continue
            break
        readings.append(dist)
    return readings

# ─── NODES ────────────────────────────────────────────────────────────────────
class OdometryNode(Node):
    """Publishes robot pose as /odom messages."""
    def __init__(self, robot_state):
        super().__init__("OdometryNode")
        self._state  = robot_state
        self._pub    = self.create_publisher("/odom")

    def step(self):
        self._pub({'x': self._state['x'], 'y': self._state['y'],
                   'theta': self._state['theta']})

class LidarNode(Node):
    """Publishes simulated 8-beam scan as /scan messages."""
    def __init__(self, robot_state):
        super().__init__("LidarNode")
        self._state = robot_state
        self._pub   = self.create_publisher("/scan")

    def step(self):
        ranges = scan_from_pose(self._state['x'], self._state['y'],
                                self._state['theta'])
        self._pub({'ranges': ranges})

class PlannerNode(Node):
    """Subscribes /odom + /scan; publishes proportional heading controller /cmd_vel."""
    def __init__(self):
        super().__init__("PlannerNode")
        self._pub   = self.create_publisher("/cmd_vel")
        self._pose  = {'x': 0.5, 'y': 0.5, 'theta': 0.0}
        self._scan  = {'ranges': [5.0]*8}
        self.create_subscription("/odom",  self._on_odom)
        self.create_subscription("/scan",  self._on_scan)

    def _on_odom(self, msg):
        self._pose = msg

    def _on_scan(self, msg):
        self._scan = msg

    def step(self):
        x, y, theta = self._pose['x'], self._pose['y'], self._pose['theta']
        goal_angle   = math.atan2(GOAL[1]-y, GOAL[0]-x)
        head_err     = ((goal_angle - theta) + math.pi) % (2*math.pi) - math.pi
        dist_to_goal = math.hypot(GOAL[0]-x, GOAL[1]-y)

        ranges = self._scan['ranges']
        front  = min(ranges[0], ranges[1], ranges[7])
        left   = min(ranges[2], ranges[3])
        right  = min(ranges[5], ranges[6])

        if front < 1.2 and dist_to_goal > 0.4:
            # Reactive avoidance: turn toward side with more open space
            angular_v = 2.5 if left >= right else -2.5
            linear_v  = 0.4
        else:
            linear_v  = 1.5 if dist_to_goal > 0.4 else 0.0
            angular_v = 2.5 * head_err

        self._pub({'linear': linear_v, 'angular': angular_v})

class MotorNode(Node):
    """Subscribes /cmd_vel; integrates differential-drive kinematics."""
    def __init__(self, robot_state, dt):
        super().__init__("MotorNode")
        self._state  = robot_state
        self._dt     = dt
        self._cmd    = {'linear': 0.0, 'angular': 0.0}
        self.create_subscription("/cmd_vel", self._on_cmd)

    def _on_cmd(self, msg):
        self._cmd = msg

    def step(self):
        v = self._cmd['linear']; w = self._cmd['angular']
        s = self._state
        noise = rng.normal(0, 0.01, 2)
        s['x']     = np.clip(s['x'] + (v*math.cos(s['theta'])+noise[0])*self._dt,
                             0, ARENA)
        s['y']     = np.clip(s['y'] + (v*math.sin(s['theta'])+noise[1])*self._dt,
                             0, ARENA)
        s['theta'] += w * self._dt
        s['theta']  = ((s['theta']+math.pi) % (2*math.pi)) - math.pi

# ─── SIMULATION LOOP ─────────────────────────────────────────────────────────
print("ROS-Style Pub/Sub Architecture Demo")
print("=" * 45)

DT       = 0.05
N_STEPS  = 500
DRAW_N   = 5

robot_state = {'x': 0.5, 'y': 0.5, 'theta': 0.0}
odom_node   = OdometryNode(robot_state)
lidar_node  = LidarNode(robot_state)
plan_node   = PlannerNode()
motor_node  = MotorNode(robot_state, DT)

path = [(robot_state['x'], robot_state['y'])]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle("ROS-Style Pub/Sub Architecture (pure Python, no ROS needed)", fontsize=10)
if INTERACTIVE:
    plt.ion()

def draw(step):
    ax.clear();  ax2.clear()
    ax.set_xlim(-0.3, ARENA+0.3);  ax.set_ylim(-0.3, ARENA+0.3)
    ax.set_aspect('equal');         ax.set_facecolor('#f5f5f5')
    done = math.hypot(robot_state['x']-GOAL[0], robot_state['y']-GOAL[1]) < 0.4
    ax.set_title(f"Arena  t={BUS.time:.1f}s  {'[GOAL]' if done else ''}", fontsize=9)
    for ox, oy, or_ in OBSTACLES:
        ax.add_patch(patches.Circle((ox,oy), or_, facecolor='tomato', alpha=0.7, zorder=2))
    pts = np.array(path)
    ax.plot(pts[:,0], pts[:,1], 'b-', alpha=0.4, lw=1, zorder=3)
    ax.plot(*GOAL, 'g*', ms=14, zorder=5, label='Goal')
    ax.plot(robot_state['x'], robot_state['y'], 'bo', ms=12, zorder=6)
    # Lidar beams
    ranges = scan_from_pose(robot_state['x'], robot_state['y'], robot_state['theta'])
    for i, r in enumerate(ranges):
        angle = robot_state['theta'] + i * (2*math.pi/8)
        ex = robot_state['x'] + math.cos(angle)*r
        ey = robot_state['y'] + math.sin(angle)*r
        ax.plot([robot_state['x'], ex], [robot_state['y'], ey],
                '-', color='cyan', alpha=0.35, lw=0.8, zorder=4)
    ax.legend(fontsize=8);  ax.grid(alpha=0.2)

    # Message log panel
    ax2.axis('off')
    ax2.set_title("Message Bus Log (last 20 messages)", fontsize=9)
    header = f"{'Time':>5}  {'Sender':<18}  {'Topic':<12}  Preview"
    rows   = [header, "-"*65]
    for t, sender, topic, preview in BUS._msg_log[-18:]:
        rows.append(f"{t:5.2f}  {sender:<18}  {topic:<12}  {preview[:30]}")
    # Node stats
    rows.append("")
    rows.append(f"{'Node':<20}  Pub counts")
    rows.append("-"*40)
    for node in [odom_node, lidar_node, plan_node, motor_node]:
        counts = "  ".join(f"{t}:{c}" for t,c in node._pub_counts.items())
        rows.append(f"{node.name:<20}  {counts}")
    ax2.text(0.02, 0.97, "\n".join(rows), transform=ax2.transAxes,
             fontsize=7, fontfamily='monospace', va='top')
    plt.tight_layout()

done = False
for step in range(N_STEPS):
    BUS.tick(DT)
    # ROS spin loop: each node's step() method publishes/receives messages
    odom_node.step()
    lidar_node.step()
    plan_node.step()
    motor_node.step()
    path.append((robot_state['x'], robot_state['y']))
    if math.hypot(robot_state['x']-GOAL[0], robot_state['y']-GOAL[1]) < 0.4:
        if not done:
            print(f"  t={BUS.time:.2f}s: Goal reached at step {step}")
            done = True
    if step % DRAW_N == 0:
        draw(step)
        if INTERACTIVE:
            plt.pause(0.03)

draw(step)
total_msgs = len(BUS._msg_log)
print(f"Simulation complete: {N_STEPS} steps, {total_msgs} messages exchanged")
print(f"Topics active: {list(BUS._subs.keys())}")

plt.savefig("ROBOTICS/ADVANCED_NVIDIA_ROS/outputs/ros_pubsub.png", dpi=90, bbox_inches='tight')
print("Saved: ROBOTICS/ADVANCED_NVIDIA_ROS/outputs/ros_pubsub.png")
if INTERACTIVE:
    plt.ioff();  plt.show()
plt.close()
