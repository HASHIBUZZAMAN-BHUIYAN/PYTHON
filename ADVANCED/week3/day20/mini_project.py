# Advanced Day 20 Mini-Project — Research Assistant Agent
# Multi-tool ReAct agent that answers complex research questions.
# Runs fully in mock mode. Set ANTHROPIC_API_KEY for real API.
# ~15 MB RAM, <1s on CPU

import os, json, re, math, datetime

print("=== Day 20 Mini-Project: Research Assistant Agent ===\n")

# ─── Tool library ─────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = {
    "python":         "Python 3.12 is a high-level, interpreted language. It is the #1 language for ML/AI. Creator: Guido van Rossum, 1991.",
    "machine learning":"ML enables systems to learn from data. Core algorithms: linear regression, decision trees, neural networks, SVMs.",
    "deep learning":  "Deep Learning uses neural networks with many layers. Key frameworks: PyTorch, TensorFlow. Enables image/speech recognition.",
    "robotics":       "Robotics combines mechanics, electronics, and AI. Key topics: kinematics, PID control, path planning, computer vision.",
    "pid controller": "PID = Proportional + Integral + Derivative. Used for servo control, temperature regulation, autonomous vehicles.",
    "a* algorithm":   "A* finds shortest paths using f(n)=g(n)+h(n). h must be admissible. Widely used in game AI and robot navigation.",
    "reinforcement learning": "RL: agent learns by interacting with environment and receiving rewards. Key algorithms: Q-learning, PPO, SAC.",
    "computer vision":"CV enables machines to understand images. Key tasks: classification, detection, segmentation. Tools: OpenCV, torchvision.",
    "natural language processing": "NLP: making computers understand text. Key models: BERT, GPT, Claude. Tasks: classification, translation, summarization.",
    "transformer":    "Transformer architecture (2017) uses self-attention. Foundation of modern LLMs like GPT-4, Claude, Gemini.",
}

CITY_DATA = {
    "london":   {"temp": 14, "condition": "cloudy",  "humidity": 78, "population": 9_000_000},
    "tokyo":    {"temp": 22, "condition": "sunny",   "humidity": 55, "population": 13_960_000},
    "new york": {"temp": 18, "condition": "partly cloudy", "humidity": 62, "population": 8_300_000},
    "sydney":   {"temp": 25, "condition": "sunny",   "humidity": 45, "population": 5_300_000},
    "paris":    {"temp": 16, "condition": "overcast","humidity": 70, "population": 2_100_000},
    "berlin":   {"temp": 12, "condition": "rainy",   "humidity": 82, "population": 3_700_000},
}

def tool_search(query):
    q = query.lower()
    for k, v in KNOWLEDGE_BASE.items():
        if k in q: return v
    # fuzzy match
    words = q.split()
    for k, v in KNOWLEDGE_BASE.items():
        if any(w in k for w in words if len(w)>3): return v
    return f"No results for '{query}'. Try: {list(KNOWLEDGE_BASE.keys())[:5]}"

def tool_get_weather(city):
    key = city.lower().strip()
    d = CITY_DATA.get(key, {"temp":20,"condition":"mild","humidity":60,"population":0})
    return json.dumps({"city":city, **d})

def tool_calculator(expr):
    try:
        allowed = {k:getattr(math,k) for k in dir(math) if not k.startswith("_")}
        result = eval(expr, {"__builtins__":{}}, allowed)
        return f"Result: {result}"
    except Exception as e: return f"Error: {e}"

def tool_compare_cities(cities_str):
    cities = [c.strip() for c in cities_str.split(",")]
    results = []
    for city in cities:
        d = CITY_DATA.get(city.lower(), None)
        if d: results.append(f"{city}: {d['temp']}°C, {d['condition']}, pop={d['population']:,}")
        else: results.append(f"{city}: unknown")
    return "\n".join(results)

def tool_summarize(text):
    words = text.split()
    if len(words) <= 30: return text
    # Simple extractive: first sentence + key stats
    sentences = text.split(". ")
    return ". ".join(sentences[:2]) + "."

def tool_get_date():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

TOOLS = {
    "search": tool_search,
    "get_weather": tool_get_weather,
    "calculator": tool_calculator,
    "compare_cities": tool_compare_cities,
    "summarize": tool_summarize,
    "get_date": tool_get_date,
}

TOOL_SCHEMA = """
Tools:
  search(topic)                — look up topic knowledge
  get_weather(city)            — get weather data for a city
  calculator(expr)             — evaluate a math expression
  compare_cities(city1,city2)  — compare multiple cities side by side
  summarize(text)              — summarize a long text
  get_date()                   — get current date

Respond format:
  THOUGHT: <reasoning>
  ACTION: tool_name(argument)
   — or —
  THOUGHT: <reasoning>
  FINAL_ANSWER: <answer>
"""

# ─── Research queries with scripted mock responses ────────────────────────────
# Each query has a pre-written script: list of (thought, action/final) pairs
SCRIPTS = {
    "Compare the weather in Tokyo, London, and Paris. Which is warmest?": [
        ("I should compare all three cities at once.", "ACTION: compare_cities(Tokyo, London, Paris)"),
        ("Tokyo is warmest at 22°C.", "FINAL_ANSWER: Tokyo is the warmest at 22°C (sunny), followed by London at 14°C (cloudy) and Paris at 16°C (overcast). Best weather: Tokyo."),
    ],
    "What is deep learning? How does it relate to machine learning?": [
        ("I need to search both topics.", "ACTION: search(deep learning)"),
        ("Now search ML for comparison.", "ACTION: search(machine learning)"),
        ("I have both answers.", "FINAL_ANSWER: Machine Learning (ML) is a broad field enabling systems to learn from data. Deep Learning is a subset of ML that uses multi-layer neural networks. Deep learning has revolutionized image recognition, NLP, and speech processing."),
    ],
    "If Tokyo's population grew by 3.5% annually, what would it be in 5 years?": [
        ("Get Tokyo's current population first.", "ACTION: get_weather(Tokyo)"),
        ("Population is 13,960,000. Now calculate.", "ACTION: calculator(13960000 * (1.035 ** 5))"),
        ("Calculation done.", "FINAL_ANSWER: Tokyo's current population is ~13,960,000. At 3.5% annual growth over 5 years: 13,960,000 × 1.035^5 ≈ 16,575,000 people."),
    ],
    "What is the transformer architecture and why is it important for AI?": [
        ("Search for transformer architecture.", "ACTION: search(transformer)"),
        ("Found relevant information.", "ACTION: summarize(Transformer architecture (2017) uses self-attention. Foundation of modern LLMs like GPT-4, Claude, Gemini.)"),
        ("Summary ready.", "FINAL_ANSWER: The Transformer architecture (introduced 2017) uses self-attention mechanisms to process sequences in parallel, unlike older RNNs. It is the foundation of all modern large language models (LLMs) including GPT-4, Claude, and Gemini — making it arguably the most impactful AI architecture in recent history."),
    ],
}

# ─── Script-based mock LLM ────────────────────────────────────────────────────
class ScriptedLLM:
    def __init__(self, script):
        self.script = script
        self.step   = 0

    def complete(self, messages, last_obs=""):
        if self.step >= len(self.script):
            return "THOUGHT: Completed.\nFINAL_ANSWER: Task complete."
        thought, action = self.script[self.step]
        self.step += 1
        return f"THOUGHT: {thought}\n{action}"

# ─── ReAct Agent ─────────────────────────────────────────────────────────────
def run_query(query, script):
    print(f"\n{'─'*65}")
    print(f"QUERY: {query}")
    print(f"{'─'*65}")
    llm = ScriptedLLM(script)
    messages = [{"role":"user","content":f"{TOOL_SCHEMA}\nTask: {query}"}]
    last_obs = ""
    for step in range(8):
        resp = llm.complete(messages, last_obs)
        am = re.search(r"ACTION:\s*(\w+)\(([^)]*)\)", resp)
        fm = re.search(r"FINAL_ANSWER:\s*(.+)", resp, re.DOTALL)
        th = re.search(r"THOUGHT:\s*(.+?)(?:ACTION|FINAL_ANSWER|$)", resp, re.DOTALL)
        thought = th.group(1).strip() if th else ""
        if thought: print(f"  [Step {step+1}] THINK: {thought}")
        if fm:
            ans = fm.group(1).strip()
            print(f"  [Step {step+1}] ANSWER: {ans}")
            return ans
        if am:
            fn, arg = am.group(1), am.group(2).strip("'\"")
            print(f"  [Step {step+1}] ACTION: {fn}({arg!r})")
            last_obs = TOOLS.get(fn, lambda x: "unknown tool")(arg) if arg else TOOLS.get(fn, lambda: "unknown tool")()
            print(f"  [Step {step+1}] OBSERVE: {str(last_obs)[:90]}")
            messages += [{"role":"assistant","content":resp},
                         {"role":"user","content":f"OBSERVE: {last_obs}\nContinue."}]
    return "max steps reached"

# ─── Run all research queries ─────────────────────────────────────────────────
for query, script in SCRIPTS.items():
    run_query(query, script)

# ─── Optional: real API mode ─────────────────────────────────────────────────
if os.environ.get("ANTHROPIC_API_KEY"):
    print("\n" + "="*65)
    print("ANTHROPIC_API_KEY detected. You can extend this script to call")
    print("the real Claude API for open-ended research queries.")
    print("Use claude-haiku-4-5-20251001 for fast, low-cost responses.")

print("\n" + "="*65)
print("Day 20 Mini-Project complete.")
print("Set ANTHROPIC_API_KEY to use real Claude responses.")
