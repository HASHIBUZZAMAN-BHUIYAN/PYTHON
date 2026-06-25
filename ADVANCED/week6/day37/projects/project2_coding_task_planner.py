"""
Project: REST API Coding Task Decomposer
Teaches: decomposing a software engineering task into ordered sub-tasks,
         ReAct-style execution with mock tool calls.
~10 MB RAM, <1s on CPU
"""
import time

# ─── Mock tools ───────────────────────────────────────────────────────────────
def tool_read_requirements(req):
    return {"endpoints": ["/users", "/posts", "/comments"],
            "auth": "JWT", "db": "SQLite", "framework": "FastAPI"}

def tool_design_schema(spec):
    return {"User": {"id":"int","email":"str","hashed_pw":"str"},
            "Post": {"id":"int","title":"str","body":"str","user_id":"int"},
            "Comment": {"id":"int","body":"str","user_id":"int","post_id":"int"}}

def tool_generate_code(spec):
    return f"# Generated: {spec}\nfrom fastapi import FastAPI\napp = FastAPI()\n# ... (stub)"

def tool_write_tests(endpoint):
    return f"def test_{endpoint.strip('/')}():\n    resp = client.get('{endpoint}')\n    assert resp.status_code == 200"

def tool_write_docs(spec):
    return f"# API Documentation\n## Endpoints\n{spec}\n## Auth: JWT Bearer token"

TOOLS = {
    "read_requirements": tool_read_requirements,
    "design_schema":     tool_design_schema,
    "generate_code":     tool_generate_code,
    "write_tests":       tool_write_tests,
    "write_docs":        tool_write_docs,
}

# ─── ReAct-style planner ──────────────────────────────────────────────────────
class ReActPlanner:
    def __init__(self, tools):
        self.tools  = tools
        self.memory = []
        self.step   = 0

    def plan(self, goal):
        plans = [
            ("read_requirements",  "Analyze the API requirements",  "Build REST API with JWT auth"),
            ("design_schema",      "Design database schema",         "User/Post/Comment models"),
            ("generate_code",      "Generate API endpoint code",     "/users endpoint"),
            ("generate_code",      "Generate POST endpoints",        "/posts endpoint"),
            ("write_tests",        "Write test cases",               "/users"),
            ("write_tests",        "Write POST test cases",          "/posts"),
            ("write_docs",         "Write API documentation",        "GET /users GET /posts"),
        ]
        return plans

    def execute(self, goal):
        print(f"Goal: {goal}\n")
        plan = self.plan(goal)
        results = {}
        for tool_name, reason, arg in plan:
            self.step += 1
            print(f"Step {self.step}: {reason}")
            print(f"  Thought: I need to {reason.lower()}")
            print(f"  Action:  {tool_name}({arg!r})")
            result = self.tools[tool_name](arg)
            print(f"  Observe: {str(result)[:80]}")
            results[reason] = result
            self.memory.append({"step": self.step, "tool":tool_name, "result":result})
            print()
            time.sleep(0)  # no actual delay; shows structure
        return results

print("=== REST API Coding Task Decomposer ===\n")
planner = ReActPlanner(TOOLS)
results = planner.execute("Build a complete CRUD REST API for a blog platform")

print("=== PLAN SUMMARY ===")
print(f"  Total steps executed: {planner.step}")
print(f"  Files generated: {sum(1 for k in results if 'code' in k.lower() or 'test' in k.lower())}")
print("\nFull memory trace:")
for m in planner.memory:
    print(f"  Step {m['step']}: {m['tool']} → {str(m['result'])[:50]}")
