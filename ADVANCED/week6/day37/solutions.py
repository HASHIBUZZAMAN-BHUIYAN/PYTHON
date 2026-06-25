# ~120 MB RAM, ~2s on CPU
"""
Day 37 — Multi-Step Planning Agents: Solutions
================================================
Full working solutions for all 5 exercises.
"""

from collections import deque

# ──────────────────────────────────────────────
# Exercise 1 — Plan-then-Execute Agent
# ──────────────────────────────────────────────
print("=" * 60)
print("EXERCISE 1 — Plan-then-Execute Agent")
print("=" * 60)

MOCK_PLANS = {
    "birthday party": [
        "Choose a date and venue",
        "Create the guest list",
        "Send invitations",
        "Plan food and cake",
        "Arrange decorations",
        "Confirm headcount and finalise",
    ],
    "job application": [
        "Update CV and cover letter",
        "Research target companies",
        "Submit applications",
        "Prepare for interviews",
        "Follow up on submissions",
    ],
    "learn python": [
        "Set up Python environment",
        "Study basic syntax (variables, loops, functions)",
        "Complete 10 exercises",
        "Build a small project",
        "Review and consolidate",
    ],
}

MOCK_RESULTS = {
    "Choose a date and venue": "Venue booked: City Hall, Saturday 15th",
    "Create the guest list": "30 guests listed",
    "Send invitations": "Invitations sent via email",
    "Plan food and cake": "Catering ordered: pizza + 3-tier cake",
    "Arrange decorations": "Balloons and banners sourced",
    "Confirm headcount and finalise": "28 confirmations received",
}


class PlanExecuteAgent:
    def plan(self, goal: str) -> list[str]:
        for key, steps in MOCK_PLANS.items():
            if key in goal.lower():
                return steps
        return ["Analyse goal", "Research options", "Take action", "Review outcome"]

    def execute(self, plan: list[str]) -> dict:
        results = {}
        for step in plan:
            results[step] = MOCK_RESULTS.get(step, f"Completed: {step}")
        return results


agent = PlanExecuteAgent()
goal = "Organise a birthday party"
print(f"\nGoal: {goal!r}")
plan = agent.plan(goal)
print(f"Plan ({len(plan)} steps):")
for i, s in enumerate(plan, 1):
    print(f"  {i}. {s}")
results = agent.execute(plan)
print("\nExecution results:")
for step, result in results.items():
    print(f"  {step[:35]:35s} → {result}")


# ──────────────────────────────────────────────
# Exercise 2 — ReAct Agent with 3 Tools
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXERCISE 2 — ReAct Agent with 3 Tools")
print("=" * 60)

DEFINITIONS = {
    "recursion": "A function that calls itself to solve smaller sub-problems.",
    "abstraction": "Hiding implementation details behind a simpler interface.",
    "polymorphism": "The ability of different types to respond to the same interface.",
    "encapsulation": "Bundling data and methods within a class.",
    "inheritance": "A class acquiring properties and methods from a parent class.",
}


class ReActAgent:
    def _search(self, query: str) -> str:
        return f"Found 3 articles about '{query}' including Wikipedia and academic papers."

    def _calculator(self, expr: str) -> str:
        try:
            # Safe: only allow simple arithmetic
            allowed = set("0123456789+-*/(). ")
            if all(c in allowed for c in expr):
                return str(eval(expr))
            return "Expression not allowed"
        except Exception as e:
            return f"Error: {e}"

    def _define(self, word: str) -> str:
        return DEFINITIONS.get(word.lower(), f"No definition found for '{word}'.")

    def run(self, question: str) -> None:
        print(f"\n  Question: {question!r}\n")
        script = [
            ("Thought", "I should search for background information first."),
            ("Action",  "search", question),
            ("Thought", "Now I'll calculate the square root of 144."),
            ("Action",  "calculator", "144 ** 0.5"),
            ("Thought", "Now I'll look up the definition of 'recursion'."),
            ("Action",  "define", "recursion"),
            ("Thought", "I have all the information I need."),
        ]
        obs = None
        for i, step in enumerate(script, 1):
            kind = step[0]
            if kind == "Thought":
                print(f"  [{i}] Thought: {step[1]}")
            else:
                tool, arg = step[1], step[2]
                result = getattr(self, f"_{tool}")(arg)
                obs = result
                print(f"  [{i}] Action : {tool}({arg!r})")
                print(f"       Observe: {obs}")


ract = ReActAgent()
ract.run("What is the square root of 144 and what does 'recursion' mean?")


# ──────────────────────────────────────────────
# Exercise 3 — Dependency Graph Builder
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXERCISE 3 — Dependency Graph Builder")
print("=" * 60)


class TaskGraph:
    def __init__(self):
        self._tasks: dict = {}  # name → set of deps

    def add_task(self, name: str, deps: list = None) -> None:
        self._tasks[name] = set(deps or [])

    def topological_order(self) -> list[str]:
        in_degree = {n: 0 for n in self._tasks}
        children = {n: [] for n in self._tasks}
        for name, deps in self._tasks.items():
            for dep in deps:
                children[dep].append(name)
                in_degree[name] += 1
        queue = deque(n for n, d in in_degree.items() if d == 0)
        order = []
        while queue:
            n = queue.popleft()
            order.append(n)
            for child in children[n]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        if len(order) != len(self._tasks):
            raise ValueError("Cycle detected!")
        return order

    def critical_path(self) -> list[str]:
        """Longest chain by number of nodes (simple DFS from each root)."""
        order = self.topological_order()
        # Build parent map
        dist = {n: 1 for n in order}
        prev = {n: None for n in order}
        children = {n: [] for n in self._tasks}
        for name, deps in self._tasks.items():
            for dep in deps:
                children[dep].append(name)
        for n in order:
            for child in children[n]:
                if dist[n] + 1 > dist[child]:
                    dist[child] = dist[n] + 1
                    prev[child] = n
        # Trace back from the node with max dist
        end = max(dist, key=lambda x: dist[x])
        path = []
        node = end
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()
        return path


tg = TaskGraph()
tg.add_task("wireframes")
tg.add_task("design",   deps=["wireframes"])
tg.add_task("backend",  deps=["wireframes"])
tg.add_task("frontend", deps=["design", "backend"])
tg.add_task("QA",       deps=["frontend"])
tg.add_task("beta",     deps=["QA"])
tg.add_task("launch",   deps=["beta"])

order = tg.topological_order()
path = tg.critical_path()
print(f"\n  Topological order: {order}")
print(f"  Critical path ({len(path)} tasks): {' → '.join(path)}")


# ──────────────────────────────────────────────
# Exercise 4 — Goal Decomposer with Depth Limit
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXERCISE 4 — Goal Decomposer with Depth Limit")
print("=" * 60)

DECOMPOSITION_TREE = {
    "Build an AI chatbot": ["Design the system", "Implement the system"],
    "Design the system":   ["Define requirements", "Choose tech stack"],
    "Implement the system":["Write NLU module", "Write dialogue flow"],
}


class GoalDecomposer:
    def decompose(self, goal: str, depth: int = 0, max_depth: int = 2) -> None:
        indent = "  " * depth
        if depth >= max_depth or goal not in DECOMPOSITION_TREE:
            print(f"{indent}- {goal}  [LEAF]")
            return
        print(f"{indent}+ {goal}")
        for sub in DECOMPOSITION_TREE[goal]:
            self.decompose(sub, depth + 1, max_depth)


gd = GoalDecomposer()
print()
gd.decompose("Build an AI chatbot")


# ──────────────────────────────────────────────
# Exercise 5 — Parallel Task Scheduler
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXERCISE 5 — Parallel Task Scheduler (wave-based)")
print("=" * 60)


class ParallelScheduler:
    def __init__(self, graph: TaskGraph):
        self._graph = graph

    def waves(self) -> list[list[str]]:
        """Return list of waves; tasks in same wave can run in parallel."""
        in_degree = {n: 0 for n in self._graph._tasks}
        children = {n: [] for n in self._graph._tasks}
        for name, deps in self._graph._tasks.items():
            for dep in deps:
                children[dep].append(name)
                in_degree[name] += 1

        result = []
        remaining = set(self._graph._tasks.keys())
        done = set()

        while remaining:
            wave = [n for n in remaining if in_degree[n] == 0]
            if not wave:
                raise ValueError("Cycle detected!")
            wave.sort()
            result.append(wave)
            for n in wave:
                remaining.remove(n)
                done.add(n)
                for child in children[n]:
                    in_degree[child] -= 1
        return result


ps = ParallelScheduler(tg)
all_waves = ps.waves()
print(f"\n  Total waves: {len(all_waves)}")
for i, wave in enumerate(all_waves, 1):
    print(f"  Wave {i}: {wave}")

print("\n--- Solutions complete ---")
