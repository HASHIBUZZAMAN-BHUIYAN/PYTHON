# ADVANCED_NVIDIA_ROS

Advanced robotics concepts modelled in pure Python: ROS pub/sub architecture,
full SLAM pipeline, imitation learning, Omniverse scene graphs, and multi-robot
DDS coordination. No GPU, no ROS installation, no cloud API needed.

---

## Honesty framing (applies to every project here)

Each script teaches **real concepts, real architecture patterns, and real code structure**
from production robotics tools. Every script also carries a 4-part docstring:

| Part | What it says |
|------|-------------|
| **REAL TOOL THIS MODELS** | Which production tool/framework this script is teaching |
| **WHAT THIS SCRIPT DOES** | Exactly what the pure-Python implementation does |
| **WHAT THE REAL TOOL ADDS** | What production requires that this script omits |
| **HARDWARE NOTE** | Confirmed to run on CPU-only modest hardware |

---

## Projects

| # | File | Models | Key concepts | Output |
|---|------|--------|-------------|--------|
| 1 | `ros_style_pubsub_architecture.py` | ROS / ROS 2 pub-sub (DDS) | MessageBus, Node, topics, spin loop; reactive obstacle avoidance planner | `ros_pubsub.png` |
| 2 | `slam_full_pipeline_sim.py` | GMapping / Cartographer / Isaac SLAM | Particle filter MCL, systematic resampling, occupancy grid map building | `slam_pipeline.png` |
| 3 | `isaac_groot_policy_concept.py` | NVIDIA Isaac GR00T (foundation model) | Behavioural cloning, NumPy MLP forward/backward, expert demo collection | `groot_policy.png` |
| 4 | `omniverse_scene_graph_concept.py` | NVIDIA Omniverse / USD | Prim tree, 4x4 homogeneous transforms, parent->world propagation, 3D arm | `omniverse_scenegraph.png` |
| 5 | `ros_navigation_stack_lite.py` | ROS Nav2 / move_base | Costmap2D with inflation, A* global planner, local planner, all as pub-sub nodes | `nav_stack.png` |
| 6 | `multi_robot_ros2_style_coordination.py` | ROS 2 multi-robot + DDS | Namespaced topics, peer-to-peer DDSDomain, fleet coordinator, 4 robot agents | `ros2_fleet.png` |

---

## Running

```
cd C:\Users\zen\Documents\GitHub\PYTHON
ROBOTICS\ADVANCED_NVIDIA_ROS\.venv\Scripts\python.exe ROBOTICS\ADVANCED_NVIDIA_ROS\<script>.py
```

All scripts detect a headless environment automatically (`MPLBACKEND=Agg`) and
save a PNG to `outputs\` without requiring a display.

---

## Environment

```
ROBOTICS\ADVANCED_NVIDIA_ROS\.venv\  (Python 3.x, CPU-only)
  numpy==2.4.x
  matplotlib==3.11.x
  scipy==1.17.x
  opencv-python==4.13.x
```

Install fresh: `python -m venv .venv && .venv\Scripts\pip install -r requirements.txt`

---

## To learn the real tools (free resources)

| Tool | Where to learn |
|------|---------------|
| ROS 2 | docs.ros.org/en/humble -- runs on Ubuntu/WSL2, free |
| Nav2 | navigation.ros.org -- move_base successor, ROS 2 |
| GMapping / Cartographer | google-cartographer.readthedocs.io |
| NVIDIA Isaac | developer.nvidia.com/isaac |
| NVIDIA Isaac GR00T | developer.nvidia.com/isaac/groot |
| NVIDIA Omniverse / USD | developer.nvidia.com/omniverse and openusd.org |

---

## Hardware note

Every project confirmed running on **Ryzen 7 5000-series CPU, 8 GB RAM, no GPU**.
Peak RAM under 150 MB per script. No internet connection required.
