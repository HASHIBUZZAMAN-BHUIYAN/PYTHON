# ~120 MB RAM, ~2s on CPU
"""
Day 37 — Multi-Step Planning Agents: Exercises
================================================
Five practice exercises.  Each is a comment block with a TODO.
"""

# ──────────────────────────────────────────────
# Exercise 1 — Plan-then-Execute Agent
# ──────────────────────────────────────────────
# TODO: Implement a PlanExecuteAgent class with:
#         plan(goal: str) -> list[str]   — mock LLM; returns 4-6 steps for the goal
#         execute(plan: list[str]) -> dict  — runs each step; returns {step: result}
#       The mock LLM should have at least 3 hardcoded goals.
#       Demonstrate with goal="Organise a birthday party".
#       Print the plan, then execute it, then print the result dict.


# ──────────────────────────────────────────────
# Exercise 2 — ReAct Agent with 3 tools
# ──────────────────────────────────────────────
# TODO: Build a ReActAgent class with tools: search, calculator, define.
#       `search(query)` returns "Found info about {query}."
#       `calculator(expr)` evaluates the expr safely (use eval on simple arithmetic).
#       `define(word)` returns a hardcoded definition for at least 5 words.
#       The agent runs a scripted ReAct loop of 6 steps for the question
#       "What is the square root of 144 and what does 'recursion' mean?".
#       Print each Thought / Action / Observe step.


# ──────────────────────────────────────────────
# Exercise 3 — Dependency Graph Builder
# ──────────────────────────────────────────────
# TODO: Write a TaskGraph class with:
#         add_task(name, deps=[])   — adds a node
#         topological_order()       — returns execution order (Kahn's algorithm)
#         critical_path()           — returns the longest chain of dependencies
#       Demonstrate with 7 tasks for "Launch a mobile app":
#         wireframes → design → backend → frontend(depends: design, backend)
#         → QA(depends: frontend) → beta(depends: QA) → launch(depends: beta)
#       Print the topological order and the critical path.


# ──────────────────────────────────────────────
# Exercise 4 — Goal Decomposer with Depth Limit
# ──────────────────────────────────────────────
# TODO: Implement a recursive GoalDecomposer.
#       decompose(goal, depth=0, max_depth=2) breaks a goal into sub-goals.
#       Mock LLM: each goal produces exactly 2 sub-goals (hardcode a tree of depth 2).
#       At max_depth, mark the task as LEAF (no further decomposition).
#       Print the full decomposition tree with indentation showing depth.
#       Root goal: "Build an AI chatbot".
#       Example tree shape:
#         Build an AI chatbot
#           Design the system
#             Define requirements  [LEAF]
#             Choose tech stack    [LEAF]
#           Implement the system
#             Write NLU module     [LEAF]
#             Write dialogue flow  [LEAF]


# ──────────────────────────────────────────────
# Exercise 5 — Parallel Task Scheduler (simulated)
# ──────────────────────────────────────────────
# TODO: Write a ParallelScheduler that takes a TaskGraph and identifies which
#       tasks can run in PARALLEL at each "wave" (all tasks whose dependencies
#       are satisfied by previous waves).
#       waves() method returns a list of lists, e.g.:
#         Wave 1: [wireframes]
#         Wave 2: [design, backend]
#         Wave 3: [frontend]
#         Wave 4: [QA]
#         ...
#       Use the same 7-task "Launch a mobile app" graph from Exercise 3.
#       Print each wave and count how many total waves are needed.
