"""
Project: 3-Day Tokyo Trip Planner Agent
Teaches: hierarchical goal decomposition, task graph, dependency-aware execution.
Fully mock — no API key required.
~10 MB RAM, <1s on CPU
"""
from collections import defaultdict, deque
import random

# ─── Mock LLM planning function ───────────────────────────────────────────────
def mock_llm(prompt):
    prompt_lower = prompt.lower()
    if "flight" in prompt_lower:
        return "Book round-trip flights to Tokyo Narita. Check JL/NH for best fares. Budget: $800."
    if "accommodation" in prompt_lower or "hotel" in prompt_lower:
        return "Book hotel in Shinjuku (3 nights). Suggested: Keio Plaza or budget capsule hotel."
    if "day 1" in prompt_lower or "arrival" in prompt_lower:
        return "Day 1: Arrive, check in, explore Shinjuku Gyoen park, dinner in Kabukicho."
    if "day 2" in prompt_lower:
        return "Day 2: Shibuya crossing, Harajuku Takeshita St, Meiji Shrine, Omotesando shops."
    if "day 3" in prompt_lower or "asakusa" in prompt_lower:
        return "Day 3: Asakusa Senso-ji, Akihabara electronics, Tokyo Skytree, farewell dinner."
    if "budget" in prompt_lower:
        return "Total budget estimate: $800 flights + $300 hotel + $300 food/activities = $1,400."
    if "packing" in prompt_lower:
        return "Pack: JR Pass, IC card, power adapter (Type A), pocket wifi, comfortable walking shoes."
    return f"Task completed: {prompt[:60]}"

# ─── Task graph ────────────────────────────────────────────────────────────────
class TaskGraph:
    def __init__(self):
        self.tasks  = {}  # id → {name, prompt, depends_on}
        self.results= {}

    def add(self, task_id, name, prompt, depends_on=None):
        self.tasks[task_id] = {"name":name, "prompt":prompt, "depends_on": depends_on or []}

    def topological_sort(self):
        in_degree = defaultdict(int)
        adj       = defaultdict(list)
        for tid, task in self.tasks.items():
            for dep in task["depends_on"]:
                adj[dep].append(tid)
                in_degree[tid] += 1
        queue   = deque(t for t in self.tasks if in_degree[t] == 0)
        ordered = []
        while queue:
            t = queue.popleft(); ordered.append(t)
            for nxt in adj[t]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0: queue.append(nxt)
        return ordered

    def execute(self, llm_fn):
        order = self.topological_sort()
        print(f"\n  Execution order: {' → '.join(order)}")
        for tid in order:
            task = self.tasks[tid]
            # Build prompt with available context from dependencies
            context = ""
            for dep in task["depends_on"]:
                if dep in self.results:
                    context += f"\n  [{dep} result]: {self.results[dep][:60]}"
            full_prompt = task["prompt"] + context
            result = llm_fn(full_prompt)
            self.results[tid] = result
            print(f"\n  [{tid}] {task['name']}")
            print(f"    → {result[:100]}")
        return self.results

# ─── Build Tokyo trip plan ─────────────────────────────────────────────────────
print("=== 3-Day Tokyo Trip Planner Agent ===\n")
print("Goal: Plan a 3-day Tokyo trip from scratch.\n")

graph = TaskGraph()
graph.add("FLIGHTS", "Book Flights",        "Book flight to Tokyo",         depends_on=[])
graph.add("HOTEL",   "Book Hotel",          "Book hotel accommodation",      depends_on=[])
graph.add("BUDGET",  "Create Budget",       "Create full trip budget plan",  depends_on=["FLIGHTS","HOTEL"])
graph.add("DAY1",    "Plan Day 1: Arrival", "Plan day 1 arrival itinerary",  depends_on=["HOTEL"])
graph.add("DAY2",    "Plan Day 2: Culture", "Plan day 2 sightseeing",        depends_on=["DAY1"])
graph.add("DAY3",    "Plan Day 3: Tokyo",   "Plan day 3 asakusa and more",   depends_on=["DAY2"])
graph.add("PACKING", "Packing List",        "Create packing list for Tokyo", depends_on=["BUDGET","DAY3"])

results = graph.execute(mock_llm)

print("\n\n=== FINAL TRIP PLAN ===")
for tid, result in results.items():
    print(f"\n[{tid}]\n  {result}")
