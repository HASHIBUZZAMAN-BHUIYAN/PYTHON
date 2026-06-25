# ~120 MB RAM, ~2s on CPU
"""
Day 37 — Multi-Step Planning Agents
=====================================
Topics:
  1. Plan-then-Execute pattern
  2. ReAct (Reason–Act–Observe) pattern
  3. Goal decomposition into sub-tasks
  4. Task dependency graph and topological ordering
  5. Mock LLM plan generation
"""

from collections import deque

# ─────────────────────────────────────────────────────────────
# SECTION 1 — PLAN-THEN-EXECUTE
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 1 — PLAN-THEN-EXECUTE PATTERN")
print("=" * 60)

"""
Plan-then-Execute:
  Phase 1: LLM (or mock) generates a COMPLETE plan as a list of steps.
  Phase 2: Executor runs each step in order.

Advantages:
  - Plan is inspectable before execution starts.
  - Easy to validate / modify before committing.

Disadvantages:
  - Plan may become stale if the environment changes mid-execution.
  - All information needed must be known at planning time.
"""


def mock_llm_plan(goal: str) -> list[str]:
    """
    Deterministic mock LLM: returns a pre-baked plan for known goals.
    Falls back to a generic 3-step plan.
    """
    plans = {
        "write a report": [
            "Gather research materials",
            "Create an outline",
            "Write a first draft",
            "Edit and proofread",
            "Finalise and submit",
        ],
        "bake a cake": [
            "Gather ingredients",
            "Preheat oven to 180°C",
            "Mix batter",
            "Pour into tin and bake 30 minutes",
            "Cool and frost",
        ],
    }
    for key, steps in plans.items():
        if key in goal.lower():
            return steps
    return ["Analyse the goal", "Break into sub-tasks", "Execute sub-tasks"]


def execute_plan(plan: list[str]) -> None:
    """Simulated step executor — just prints what it would do."""
    print(f"  Plan has {len(plan)} steps:")
    for i, step in enumerate(plan, 1):
        print(f"    Step {i}: {step}")
        # In a real agent: tool calls, API requests, etc.
    print("  [Plan execution complete]")


goal = "Write a report on climate change"
print(f"\nGoal: {goal!r}")
plan = mock_llm_plan(goal)
execute_plan(plan)


# ─────────────────────────────────────────────────────────────
# SECTION 2 — REACT PATTERN (REASON-ACT-OBSERVE)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2 — REACT PATTERN")
print("=" * 60)

"""
ReAct interleaves:
  Thought  — internal reasoning
  Action   — tool call or operation
  Observe  — result of the action

The agent decides the NEXT step based on the CURRENT observation.
This is more adaptive than plan-then-execute.
"""

MOCK_TOOLS = {
    "search": lambda q: f"[Search result for '{q}': Found 3 relevant articles]",
    "calculator": lambda expr: f"[Calc: {expr} = {eval(expr)}]",
    "summarise": lambda text: f"[Summary: {text[:40]}...]",
}


def react_agent(question: str, max_steps: int = 4) -> str:
    """
    Simple ReAct agent with a scripted trace (mock LLM decisions).
    """
    script = [
        ("Thought", "I need to search for information about this topic."),
        ("Action",  "search", question),
        ("Thought", "I found some articles. Let me summarise the key findings."),
        ("Action",  "summarise", f"Articles about {question}"),
    ]

    print(f"\n  Question: {question!r}\n")
    observation = None

    for i, step in enumerate(script[:max_steps], 1):
        kind = step[0]
        if kind == "Thought":
            print(f"  [{i}] Thought: {step[1]}")
        elif kind == "Action":
            tool_name = step[1]
            arg = step[2]
            if observation:
                arg = observation  # chain: use last observation
            if tool_name in MOCK_TOOLS:
                observation = MOCK_TOOLS[tool_name](arg)
                print(f"  [{i}] Action: {tool_name}({arg!r})")
                print(f"       Observe: {observation}")
            else:
                print(f"  [{i}] Action: {tool_name} — tool not found")

    return f"Final answer based on research about {question}."


answer = react_agent("the water cycle")
print(f"\n  Final: {answer}")


# ─────────────────────────────────────────────────────────────
# SECTION 3 — GOAL DECOMPOSITION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3 — GOAL DECOMPOSITION")
print("=" * 60)

"""
Goal decomposition turns a high-level goal into a tree of sub-goals.
Each sub-goal can be further decomposed until it's a concrete action.

Example tree:
  "Build a web app"
    ├── "Design database schema"
    ├── "Create backend API"
    │     ├── "Define endpoints"
    │     └── "Add authentication"
    └── "Build frontend"
"""


class SubTask:
    def __init__(self, name: str, deps: list = None):
        self.name = name
        self.deps = deps or []   # list of SubTask names that must complete first

    def __repr__(self):
        return f"SubTask({self.name!r}, deps={self.deps})"


def decompose_goal(goal: str) -> list[SubTask]:
    """Mock LLM decomposition — returns hardcoded sub-tasks for known goals."""
    if "web app" in goal.lower():
        return [
            SubTask("design_schema"),
            SubTask("define_endpoints",   deps=["design_schema"]),
            SubTask("add_auth",            deps=["define_endpoints"]),
            SubTask("build_frontend",      deps=["design_schema"]),
            SubTask("integration_tests",   deps=["add_auth", "build_frontend"]),
            SubTask("deploy",              deps=["integration_tests"]),
        ]
    return [SubTask("analyse"), SubTask("implement", deps=["analyse"]), SubTask("test", deps=["implement"])]


tasks = decompose_goal("Build a web app")
print(f"\n  Decomposed {len(tasks)} sub-tasks:")
for t in tasks:
    print(f"    {t}")


# ─────────────────────────────────────────────────────────────
# SECTION 4 — TOPOLOGICAL ORDERING (Kahn's algorithm)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 4 — TOPOLOGICAL ORDERING")
print("=" * 60)


def topological_sort(tasks: list[SubTask]) -> list[str]:
    """
    Kahn's algorithm: BFS-based topological sort.
    Returns task names in execution order.
    Raises ValueError on cycle.
    """
    # Build adjacency and in-degree maps
    task_names = {t.name for t in tasks}
    in_degree = {t.name: 0 for t in tasks}
    dependents = {t.name: [] for t in tasks}  # dep → tasks that need it

    for t in tasks:
        for dep in t.deps:
            if dep not in task_names:
                raise ValueError(f"Unknown dependency: {dep!r}")
            dependents[dep].append(t.name)
            in_degree[t.name] += 1

    queue = deque([name for name, deg in in_degree.items() if deg == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for child in dependents[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(order) != len(tasks):
        raise ValueError("Cycle detected in task dependency graph!")

    return order


order = topological_sort(tasks)
print("\n  Execution order after topological sort:")
for i, name in enumerate(order, 1):
    print(f"    {i}. {name}")


# ─────────────────────────────────────────────────────────────
# SECTION 5 — FULL PLANNING PIPELINE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 5 — FULL PLANNING PIPELINE")
print("=" * 60)

"""
Full pipeline:
  1. User provides goal
  2. Mock LLM decomposes into sub-tasks with dependencies
  3. Topological sort determines safe execution order
  4. Executor runs each task (mock)
"""


def mock_execute_task(task_name: str, context: dict) -> str:
    """Returns a mock result string for a named task."""
    results = {
        "design_schema":      "Schema: users, posts, comments tables defined.",
        "define_endpoints":   "Endpoints: GET /posts, POST /posts, DELETE /posts/:id",
        "add_auth":           "JWT authentication middleware added.",
        "build_frontend":     "React app scaffolded with Vite.",
        "integration_tests":  "12 tests passed, 0 failed.",
        "deploy":             "Docker image built and pushed to registry.",
    }
    return results.get(task_name, f"Task {task_name!r} completed.")


def run_plan(goal: str) -> None:
    print(f"\n  Goal: {goal!r}")
    sub_tasks = decompose_goal(goal)
    order = topological_sort(sub_tasks)
    context = {}
    print(f"  Execution plan ({len(order)} steps):")
    for i, name in enumerate(order, 1):
        result = mock_execute_task(name, context)
        context[name] = result
        print(f"    [{i}] {name:25s} → {result}")
    print("  [Goal achieved]\n")


run_plan("Build a web app")
print("\n--- Lesson complete ---")
