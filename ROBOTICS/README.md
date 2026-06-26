# ROBOTICS Reference Folder

Reusable robotics primitives and real-time simulation projects.

| File | Contents |
|------|----------|
| `pid_controller.py`    | PID class with anti-windup, cascade PID, 2D PID |
| `pathfinding_astar.py` | A* (4 & 8-connected), BFS, Dijkstra, occupancy grid helpers |

## Environment Setup

This folder has its **own dedicated virtual environment** (`ROBOTICS\.venv`) -- separate from every other folder including BEGINNER and ADVANCED.

**From a fresh terminal:**
```
cd C:\Users\zen\Documents\GitHub\PYTHON
ROBOTICS\.venv\Scripts\activate
python ROBOTICS\pathfinding_astar.py
```

**Or:** double-click `ROBOTICS\activate.bat` -- it activates the venv and sets the working directory automatically.

Installed packages (see `ROBOTICS\requirements.txt`): numpy, matplotlib, scipy, opencv-python

---

## Related lessons
- ADVANCED/week3/day15 -- Robot kinematics
- ADVANCED/week3/day16 -- PID controller
- ADVANCED/week3/day17 -- Path planning
- ADVANCED/week3/day18 -- OpenCV vision

---

## Round 2: Real-Time Simulation Projects

All 10 projects are real-time matplotlib animations.
- Run from `C:\Users\zen\Documents\GitHub\PYTHON\` with the ROBOTICS venv active
- Each saves a final PNG to `ROBOTICS\outputs\`
- Set `MPLBACKEND=Agg` to skip display (headless/CI testing)

> **Hardware:** Ryzen 7 5000-series, 8 GB RAM, No GPU -- all CPU-only.
> Grid sizes: up to 22x22. Max 6 robots. Run times: 5-15s with animation, <2s headless.

### Mobile Navigation

| File | Description | Algorithm | Run command |
|------|-------------|-----------|-------------|
| `realtime_obstacle_avoidance.py` | Robot navigates 10x10 m arena with 6 circular obstacles. Left panel: robot path + force arrows. Right panel: live force decomposition (F_att, F_rep, F_total). | Potential Fields (Khatib 1986) + position-progress stuck detection + escape mode | `python ROBOTICS\realtime_obstacle_avoidance.py` |
| `slam_lite_mapping.py` | Robot builds an occupancy grid map of a 22x22 maze using simulated 8-directional range sensors. Side-by-side: ground truth vs. built map. | Ray-casting + log-odds occupancy update + uncertainty-driven exploration | `python ROBOTICS\slam_lite_mapping.py` |
| `multi_goal_dynamic_routing.py` | Robot visits 6 goals; a new obstacle appears at step 180, triggering live tour re-planning. | Nearest-Neighbor TSP heuristic + potential-field local navigation + online replanning | `python ROBOTICS\multi_goal_dynamic_routing.py` |

### Manipulation

| File | Description | Algorithm | Run command |
|------|-------------|-----------|-------------|
| `inverse_kinematics_arm.py` | 2-link planar arm (L1=3.5, L2=2.5 m) moves through 6 targets. Left: workspace + trajectory. Right: joint angle history (theta1, theta2). | Analytical closed-form IK + cosine ease-in/ease-out interpolation | `python ROBOTICS\inverse_kinematics_arm.py` |
| `pick_and_place_simulation.py` | Arm picks 4 coloured boxes and places them at a drop zone. Objects visually attach to end-effector when grasped. | FSM (MOVE_TO_OBJ -> GRASP -> LIFT -> MOVE_TO_DROP -> PLACE) + IK | `python ROBOTICS\pick_and_place_simulation.py` |
| `trajectory_planning_smooth.py` | Side-by-side comparison of 3 trajectory types through 5 waypoints: linear, cubic polynomial, quintic polynomial. Shows speed + acceleration profiles. | Cubic/quintic boundary-condition polynomial + animated speed/acceleration plots | `python ROBOTICS\trajectory_planning_smooth.py` |

### Swarm

| File | Description | Algorithm | Run command |
|------|-------------|-----------|-------------|
| `formation_control_swarm.py` | 5 robots in V-formation follow a figure-8 target. Right panel shows formation error over time. | Virtual Structure + PD control + inter-robot repulsion | `python ROBOTICS\formation_control_swarm.py` |
| `multi_robot_task_allocation.py` | 3 robots, 7 tasks. Greedy auction assigns tasks dynamically as robots become free. Assignment log shown live. | Greedy auction (minimum-distance bidding) + real-time reassignment | `python ROBOTICS\multi_robot_task_allocation.py` |
| `swarm_collision_avoidance.py` | 6 robots cross from one side of a circle to the opposite (classic ORCA benchmark). Right panel: minimum pairwise distance over time. | ORCA-lite (CPA-based VO + right-hand traffic rule) | `python ROBOTICS\swarm_collision_avoidance.py` |

### Webcam Bonus

| File | Description | Mode | Run command |
|------|-------------|------|-------------|
| `webcam_robot_teleop_sim.py` | Tracks a green object via webcam and drives a simulated 2D robot. If no webcam: automatic synthetic Lissajous fallback -- always works without hardware. | WEBCAM: HSV threshold + proportional control. FALLBACK: figure-8 synthetic input | `python ROBOTICS\webcam_robot_teleop_sim.py` |

> **Outputs:** all saved to `ROBOTICS\outputs\` as PNG files after each run.
> `obstacle_avoidance.png`, `slam_map.png`, `multi_goal_routing.png`, `ik_arm.png`,
> `pick_and_place.png`, `trajectory_planning.png`, `formation_control.png`,
> `task_allocation.png`, `swarm_avoidance.png`, `webcam_teleop.png`

---

## ADVANCED_NVIDIA_ROS Subfolder

See [`ADVANCED_NVIDIA_ROS/README.md`](ADVANCED_NVIDIA_ROS/README.md) for 6 advanced projects
modelling real production robotics systems (ROS pub-sub, full SLAM pipeline,
NVIDIA Isaac GR00T imitation learning, Omniverse USD scene graphs, Nav2 nav stack,
and ROS 2 DDS multi-robot fleet). Each has its own venv and runs CPU-only.

| # | Script | Models |
|---|--------|--------|
| 1 | `ros_style_pubsub_architecture.py` | ROS / ROS 2 pub-sub, node graph |
| 2 | `slam_full_pipeline_sim.py` | Particle filter SLAM (GMapping/Cartographer) |
| 3 | `isaac_groot_policy_concept.py` | NVIDIA Isaac GR00T imitation learning |
| 4 | `omniverse_scene_graph_concept.py` | Omniverse USD scene graph, 3D transforms |
| 5 | `ros_navigation_stack_lite.py` | Nav2 / move_base: Costmap + A* + local planner |
| 6 | `multi_robot_ros2_style_coordination.py` | ROS 2 DDS multi-robot fleet (4 robots) |
