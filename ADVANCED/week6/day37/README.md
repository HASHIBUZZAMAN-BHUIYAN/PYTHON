# Day 37 — Multi-Step Planning Agents

## Learning Objectives
- Understand the Plan-then-Execute vs ReAct (interleaved) patterns
- Decompose a high-level goal into sub-tasks with a mock LLM
- Model task dependencies as a directed graph and run topological sort
- Execute a plan step-by-step, passing outputs between steps

## Topics Covered
1. **Plan-then-Execute** — generate the entire plan first, then execute
2. **ReAct pattern** — interleave Reason / Act / Observe steps
3. **Goal Decomposition** — break a goal into an ordered list of sub-tasks
4. **Task Dependency Graph** — topological ordering of dependent sub-tasks
5. **Mock LLM Planning** — deterministic plan generation without an API key

## Files
| File | Description |
|------|-------------|
| `lesson.py` | Full walkthrough of planning patterns |
| `exercises.py` | 5 practice TODOs |
| `solutions.py` | Complete solutions |
| `projects/project1_trip_planner.py` | 3-day Tokyo trip planner agent |
| `projects/project2_coding_task_planner.py` | REST API coding task decomposer |
| `projects/project3_research_planner.py` | Research question decomposition agent |

## Hardware
- CPU-only (no GPU), no API key required

## Run
```
python lesson.py
python projects/project1_trip_planner.py
python projects/project2_coding_task_planner.py
python projects/project3_research_planner.py
```
