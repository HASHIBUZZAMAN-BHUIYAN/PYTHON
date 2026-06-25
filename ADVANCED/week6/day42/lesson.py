# Advanced Day 42 — Final Capstone Lesson: Putting It All Together
# ~30 MB RAM, ~2s on CPU

print("""
=== Final Capstone — Day 42 ===

You have completed 42 days of Python AI/ML curriculum.
This lesson reviews the key concepts and shows how they connect.

  [NLP]  + [DL]  + [Agents]  =  End-to-end AI System

The architecture of a modern AI assistant:
─────────────────────────────────────────────────────────
  User Input → [Input Guardrail] → [NLP Understanding]
                                         ↓
                              [Agent Planning Layer]
                                    ↙          ↘
                          [Tool Execution]  [Memory/RAG]
                                    ↘          ↙
                              [Response Generation]
                                         ↓
                    [Output Guardrail] → User Output
─────────────────────────────────────────────────────────
""")

import re, time
from collections import Counter

# ─── COMPONENT 1: NLP Understanding ─────────────────────────────────────────
print("=== 1. NLP: Intent Classification ===")

INTENTS = {
    "greeting":    ["hello","hi","hey","good morning","greetings"],
    "farewell":    ["bye","goodbye","see you","later","cya"],
    "weather":     ["weather","temperature","rain","sunny","forecast"],
    "calculation": ["calculate","compute","what is","how much","add","multiply","divide"],
    "search":      ["find","search","look up","what is","who is","tell me about"],
    "help":        ["help","how do i","how to","what can you","commands"],
}

def classify_intent(text):
    text_lower = text.lower()
    scores = {}
    for intent, keywords in INTENTS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score: scores[intent] = score
    if not scores: return "unknown", 0.0
    best = max(scores, key=scores.get)
    total = sum(scores.values())
    return best, scores[best] / total

def extract_entities(text):
    entities = {}
    nums = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    if nums: entities["numbers"] = [float(n) for n in nums]
    ops = re.findall(r"\b(add|plus|subtract|minus|multiply|times|divide|plus)\b", text.lower())
    if ops: entities["operations"] = ops
    return entities

queries = [
    "Hello there!",
    "What is 42 plus 58?",
    "Search for machine learning tutorials",
    "What's the weather like today?",
    "How do I get started?",
    "Calculate 15 times 4",
]
print(f"  {'Query':<40}  {'Intent':<15}  Confidence")
print("  " + "─"*65)
for q in queries:
    intent, conf = classify_intent(q)
    ents = extract_entities(q)
    ent_str = str(ents) if ents else ""
    print(f"  {q:<40}  {intent:<15}  {conf:.0%}  {ent_str}")

# ─── COMPONENT 2: Tool Registry ──────────────────────────────────────────────
print("\n=== 2. Agent: Tool Registry ===")

class ToolRegistry:
    def __init__(self):
        self._tools = {}
    def register(self, name, fn, description=""):
        self._tools[name] = {"fn": fn, "desc": description}
        return fn
    def call(self, name, **kwargs):
        if name not in self._tools: return f"Error: tool '{name}' not found"
        try: return self._tools[name]["fn"](**kwargs)
        except Exception as e: return f"Error: {e}"
    def list_tools(self): return list(self._tools.keys())

registry = ToolRegistry()
registry.register("add",         lambda a,b: a+b,           "Add two numbers")
registry.register("multiply",    lambda a,b: a*b,           "Multiply two numbers")
registry.register("divide",      lambda a,b: a/b if b!=0 else "err:div0", "Divide a by b")
registry.register("get_weather", lambda city: f"Mock: {city} is 22°C, partly cloudy", "Get weather")
registry.register("search",      lambda query: f"Mock results for: {query[:30]}", "Search the web")

print(f"  Registered tools: {registry.list_tools()}")
print(f"  add(3,4) = {registry.call('add', a=3, b=4)}")
print(f"  multiply(7,6) = {registry.call('multiply', a=7, b=6)}")
print(f"  get_weather('Paris') = {registry.call('get_weather', city='Paris')}")

# ─── COMPONENT 3: Simple Planning Loop ───────────────────────────────────────
print("\n=== 3. Agent: Sense-Plan-Act Loop ===")

def agent_loop(query, registry, max_steps=3):
    print(f"\n  Query: '{query}'")
    intent, conf = classify_intent(query)
    entities     = extract_entities(query)
    print(f"  Intent: {intent} ({conf:.0%})  Entities: {entities}")

    for step in range(max_steps):
        if intent == "calculation" and "numbers" in entities:
            nums = entities["numbers"]
            ops  = entities.get("operations", ["add"])
            op   = ops[0] if ops else "add"
            fn_map = {"add":"add","plus":"add","multiply":"multiply","times":"multiply","divide":"divide"}
            tool = fn_map.get(op, "add")
            if len(nums) >= 2:
                result = registry.call(tool, a=nums[0], b=nums[1])
                return f"Result: {nums[0]} {op} {nums[1]} = {result}"
        elif intent == "weather":
            return registry.call("get_weather", city="your location")
        elif intent == "search":
            topic = re.sub(r"search|find|look up|tell me about", "", query.lower()).strip()
            return registry.call("search", query=topic)
        elif intent == "greeting":
            return "Hello! I'm your AI assistant. How can I help you today?"
        elif intent == "help":
            return f"I can help with: {', '.join(registry.list_tools())}. Just ask!"
        else:
            return "I'm not sure how to help with that. Try asking about weather, calculations, or searching."
    return "Max steps reached. Please rephrase your query."

for q in queries[:4]:
    answer = agent_loop(q, registry)
    print(f"  → {answer}")

print("""
=== Day 42 Summary: What You've Built ===

  6 weeks × 7 days = 42 lessons + 42 exercises + 42 solution sets
  + 3 projects per day from Week 4-6 = ~63 project files

  Core concepts mastered:
    ✓ NumPy, Pandas, Matplotlib data science
    ✓ Scikit-learn ML pipelines and evaluation
    ✓ PyTorch deep learning (MLP, CNN, RNN, LSTM, GRU)
    ✓ NLP: sentiment, NER, text classification, generation
    ✓ Autoencoders, GANs, Attention, Transformers
    ✓ Agentic AI: tool use, planning, multi-agent, guardrails

Congratulations! You now have a solid foundation in Python AI/ML.
""")
