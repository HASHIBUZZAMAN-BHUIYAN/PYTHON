# Advanced Day 21 — CAPSTONE: Full Agentic Robotic System

## Objectives
- Integrate all Week 3 skills into one system
- Agent loop: perceive (OpenCV) → plan (A*) → control (PID) → act
- Grid world environment with colored goal beacons
- LLM decision layer selects mission goals (mock mode, no API required)
- Visualize the full mission with matplotlib

## System Architecture
```
LLM Agent (Day 20)
    ↓ "Go to blue beacon, then red"
FSM Controller (Day 19)
    ↓ state transitions
A* Path Planner (Day 17)
    ↓ waypoints
PID Follower (Day 16)
    ↓ steering commands
Robot Simulator (Day 15)
    ↓ x, y, θ
OpenCV Vision (Day 18)
    ↓ beacon detection
```

## How to run
```powershell
python ADVANCED\week3\day21\lesson.py
python ADVANCED\week3\day21\mini_project.py
```

## Time estimate
~90 minutes
