# Advanced Day 20 — LLM-Powered Agents
# ~15 MB RAM, <1s on CPU (mock mode)
# OPTIONAL: set ANTHROPIC_API_KEY for real LLM calls

import os, json, re, math, datetime

print("""
=== LLM-Powered Agents — Day 20 ===

Architecture: ReAct (Reason + Act) loop
  1. THINK  — LLM reasons about the task
  2. ACT    — LLM selects a tool and arguments
  3. OBSERVE— Tool executes, result returned to LLM
  4. LOOP   — Repeat until LLM says FINAL_ANSWER

Tools available to the agent:
  calculator(expr)         — evaluate a math expression
  get_weather(city)        — mock weather data
  search(query)            — mock web search
  read_file(path)          — mock file contents
  get_date()               — current date/time

If ANTHROPIC_API_KEY is set, uses real Claude API.
Otherwise, uses a deterministic mock LLM.
""")

# ─── Tool definitions ─────────────────────────────────────────────────────────
def tool_calculator(expr: str) -> str:
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expr, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

def tool_get_weather(city: str) -> str:
    mock_data = {
        "london":  {"temp": 14, "condition": "cloudy",  "humidity": 78},
        "tokyo":   {"temp": 22, "condition": "sunny",   "humidity": 55},
        "new york":{"temp": 18, "condition": "partly cloudy", "humidity": 62},
        "sydney":  {"temp": 25, "condition": "sunny",   "humidity": 45},
    }
    key = city.lower().strip()
    data = mock_data.get(key, {"temp": 20, "condition": "mild", "humidity": 60})
    return json.dumps({"city": city, **data})

def tool_search(query: str) -> str:
    mock_results = {
        "python": "Python is a high-level programming language known for readability. Latest version: 3.12.",
        "machine learning": "ML is a subset of AI that enables systems to learn from data without explicit programming.",
        "pid controller": "A PID controller is a control loop mechanism using Proportional, Integral, Derivative terms.",
        "astar": "A* is a graph traversal algorithm that finds the shortest path using a heuristic function.",
    }
    for k, v in mock_results.items():
        if k in query.lower():
            return v
    return f"Search results for '{query}': (mock) No specific results found. Try a more specific query."

def tool_read_file(path: str) -> str:
    mock_files = {
        "data.txt":   "Sales Q1: 12000\nSales Q2: 15000\nSales Q3: 18000\nSales Q4: 21000",
        "config.json": '{"model": "gpt-4", "temperature": 0.7, "max_tokens": 2048}',
        "notes.md":   "# Meeting Notes\n- Discussed Q4 targets\n- Action: improve ML pipeline\n- Next meeting: next Monday",
    }
    return mock_files.get(os.path.basename(path), f"File '{path}' not found (mock mode).")

def tool_get_date() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

TOOLS = {
    "calculator": tool_calculator,
    "get_weather": tool_get_weather,
    "search": tool_search,
    "read_file": tool_read_file,
    "get_date": tool_get_date,
}

TOOL_DESCRIPTIONS = """
Available tools:
  calculator(expr)     — Evaluate a Python math expression. Example: calculator("2**10 + sqrt(9)")
  get_weather(city)    — Get current weather for a city. Example: get_weather("London")
  search(query)        — Search for information. Example: search("what is machine learning")
  read_file(path)      — Read a file. Example: read_file("data.txt")
  get_date()           — Get current date and time.

To use a tool, respond EXACTLY in this format:
  THOUGHT: <your reasoning>
  ACTION: tool_name(argument)

When you have the final answer:
  THOUGHT: <your reasoning>
  FINAL_ANSWER: <your answer>
"""

# ─── Mock LLM ─────────────────────────────────────────────────────────────────
class MockLLM:
    """Deterministic mock that handles a fixed set of query patterns."""

    def __init__(self):
        self.rules = [
            # (regex on full context, response template)
            (r"what.*weather.*in\s+(\w[\w\s]*?)[\?\.]",
             lambda m: f"THOUGHT: I need to check the weather.\nACTION: get_weather({m.group(1).strip()})"),
            (r"calculator.*result.*OBSERVE",
             lambda m: "THOUGHT: I have the calculation result.\nFINAL_ANSWER: {last_obs}"),
            (r"calculate|compute|what is\s+([\d\s\+\-\*/\^\.]+)",
             lambda m: f"THOUGHT: I need to calculate this.\nACTION: calculator({m.group(1).strip()})"),
            (r"search.*OBSERVE",
             lambda m: "THOUGHT: I found relevant information.\nFINAL_ANSWER: {last_obs}"),
            (r"find.*information|tell me about|what is\s+([a-z][\w\s]+)",
             lambda m: f"THOUGHT: I should search for this.\nACTION: search({m.group(1).strip()})"),
            (r"read.*file|open.*file",
             lambda m: "THOUGHT: I need to read the file.\nACTION: read_file(data.txt)"),
            (r"read_file.*OBSERVE",
             lambda m: "THOUGHT: I have the file contents.\nFINAL_ANSWER: {last_obs}"),
            (r"today|current date|what.*date|what time",
             lambda m: "THOUGHT: I need the current date.\nACTION: get_date()"),
            (r"get_date.*OBSERVE",
             lambda m: "THOUGHT: I have the date.\nFINAL_ANSWER: {last_obs}"),
            (r"weather.*OBSERVE",
             lambda m: "THOUGHT: I have the weather data.\nFINAL_ANSWER: {last_obs}"),
        ]

    def complete(self, messages: list, last_obs: str = "") -> str:
        full_text = " ".join(m.get("content","") for m in messages).lower()
        for pattern, response_fn in self.rules:
            m = re.search(pattern, full_text, re.IGNORECASE)
            if m:
                resp = response_fn(m)
                return resp.replace("{last_obs}", last_obs)
        return f"THOUGHT: I'll answer directly.\nFINAL_ANSWER: Based on the context, here is my response to your query."

# ─── Real LLM (optional) ──────────────────────────────────────────────────────
class AnthropicLLM:
    def __init__(self):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            self.available = True
        except Exception:
            self.available = False

    def complete(self, messages: list, last_obs: str = "") -> str:
        system = f"You are a helpful AI agent.\n{TOOL_DESCRIPTIONS}"
        try:
            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                system=system,
                messages=messages,
            )
            return response.content[0].text
        except Exception as e:
            return f"THOUGHT: API error: {e}\nFINAL_ANSWER: Unable to complete request."

# ─── ReAct Agent ─────────────────────────────────────────────────────────────
class ReActAgent:
    def __init__(self, llm, max_steps=6, verbose=True):
        self.llm       = llm
        self.max_steps = max_steps
        self.verbose   = verbose

    def parse_response(self, text):
        # Extract ACTION or FINAL_ANSWER
        action_match = re.search(r"ACTION:\s*(\w+)\(([^)]*)\)", text, re.IGNORECASE)
        final_match  = re.search(r"FINAL_ANSWER:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
        thought_match= re.search(r"THOUGHT:\s*(.+?)(?:ACTION|FINAL_ANSWER|$)", text, re.DOTALL | re.IGNORECASE)

        thought = thought_match.group(1).strip() if thought_match else ""
        if final_match:
            return "final", final_match.group(1).strip(), thought
        if action_match:
            return "action", (action_match.group(1), action_match.group(2).strip("'\"")), thought
        return "unknown", text, ""

    def run(self, query):
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")

        messages = [{"role": "user", "content": f"{TOOL_DESCRIPTIONS}\n\nTask: {query}"}]
        last_obs = ""

        for step in range(self.max_steps):
            response = self.llm.complete(messages, last_obs)
            kind, content, thought = self.parse_response(response)

            if self.verbose and thought:
                print(f"\n[Step {step+1}] THOUGHT: {thought[:100]}...")

            if kind == "final":
                print(f"\n[Step {step+1}] FINAL ANSWER: {content}")
                return content

            if kind == "action":
                tool_name, tool_arg = content
                if self.verbose:
                    print(f"[Step {step+1}] ACTION: {tool_name}({tool_arg!r})")
                # Execute tool
                if tool_name in TOOLS:
                    if tool_arg:
                        obs = TOOLS[tool_name](tool_arg)
                    else:
                        obs = TOOLS[tool_name]()
                else:
                    obs = f"Unknown tool: {tool_name}"
                last_obs = obs
                if self.verbose:
                    print(f"[Step {step+1}] OBSERVE: {obs[:100]}")
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user",      "content": f"OBSERVE: {obs}\n\nContinue."})
            else:
                print(f"[Step {step+1}] (unexpected format) {content[:80]}")
                break

        return "Max steps reached."

# ─── Setup LLM ────────────────────────────────────────────────────────────────
api_key = os.environ.get("ANTHROPIC_API_KEY")
if api_key:
    llm_real = AnthropicLLM()
    if llm_real.available:
        llm = llm_real
        print("Using real Anthropic API (claude-haiku-4-5-20251001)")
    else:
        llm = MockLLM()
        print("API key set but import failed — using mock LLM")
else:
    llm = MockLLM()
    print("No API key found — running in MOCK MODE (deterministic responses)\n")

agent = ReActAgent(llm, max_steps=6, verbose=True)

# ─── Demo queries ─────────────────────────────────────────────────────────────
queries = [
    "What is the weather in Tokyo?",
    "What is today's date and time?",
    "Tell me about machine learning",
    "Calculate 2**10 + sqrt(144)",
]
for q in queries:
    agent.run(q)

print("\n" + "="*60)
print("Day 20 lesson complete.")
print("Set ANTHROPIC_API_KEY to use real Claude API responses.")
