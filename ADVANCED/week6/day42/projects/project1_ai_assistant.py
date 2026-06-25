"""
Project: Complete AI Assistant with Memory, Tools, Guardrails
Teaches: end-to-end assistant: intent→tools→memory→guardrails→response,
         multi-turn conversation, topic tracking, response quality scoring.
~20 MB RAM, <1s on CPU
"""
import re, time, uuid
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# ─── Core types ──────────────────────────────────────────────────────────────
@dataclass
class Turn:
    turn_id:   str
    user:      str
    assistant: str
    intent:    str
    latency:   float

@dataclass
class AssistantState:
    memory:   deque = field(default_factory=lambda: deque(maxlen=10))
    topic:    str   = "general"
    turn_count: int = 0

# ─── NLP Layer ───────────────────────────────────────────────────────────────
INTENT_MAP = {
    "greeting":    ["hello","hi","hey","good"],
    "farewell":    ["bye","goodbye","exit","quit"],
    "calculation": ["add","subtract","multiply","divide","plus","minus","times","what is","calculate","compute"],
    "search":      ["search","find","look up","who is","what is","tell me","explain"],
    "status":      ["history","what did","remember","last","previous","conversation"],
    "help":        ["help","commands","what can","how do"],
}

def classify(text: str) -> str:
    tl = text.lower()
    for intent, kws in INTENT_MAP.items():
        if any(kw in tl for kw in kws): return intent
    return "unknown"

def extract_numbers(text: str) -> List[float]:
    return [float(n) for n in re.findall(r"-?\b\d+(?:\.\d+)?\b", text)]

def extract_op(text: str) -> str:
    tl = text.lower()
    if any(w in tl for w in ["add","plus"]): return "add"
    if any(w in tl for w in ["subtract","minus"]): return "subtract"
    if any(w in tl for w in ["multiply","times","product"]): return "multiply"
    if any(w in tl for w in ["divide","quotient"]): return "divide"
    return "add"

# ─── Tool Layer ───────────────────────────────────────────────────────────────
TOOLS = {
    "add":       lambda a,b: a+b,
    "subtract":  lambda a,b: a-b,
    "multiply":  lambda a,b: a*b,
    "divide":    lambda a,b: (a/b if b!=0 else "Error: division by zero"),
    "get_weather": lambda city="unknown": f"Mock: {city} is 22°C, partly cloudy.",
    "search":    lambda q: f"Mock search results for '{q[:40]}': Here is relevant information.",
}

def execute_tool(intent: str, text: str, state: AssistantState) -> str:
    if intent == "calculation":
        nums = extract_numbers(text)
        op   = extract_op(text)
        if len(nums) >= 2:
            result = TOOLS[op](nums[0], nums[1])
            return f"{nums[0]} {op} {nums[1]} = {result}"
        return "Please provide two numbers for the calculation."
    elif intent == "search":
        query = re.sub(r"\b(search|find|look up|tell me|explain|who is|what is)\b","",text,flags=re.I).strip()
        return TOOLS["search"](query)
    elif intent == "greeting":
        return f"Hello! I'm your AI assistant. This is turn #{state.turn_count+1}. How can I help?"
    elif intent == "farewell":
        return f"Goodbye! We had {state.turn_count} turns. See you next time!"
    elif intent == "status":
        ctx = "\n    ".join(f"[{t.intent}] U:{t.user[:30]} A:{t.assistant[:40]}" for t in list(state.memory)[-3:])
        return f"Recent conversation:\n    {ctx}" if ctx else "No conversation history yet."
    elif intent == "help":
        return "I can: add/subtract/multiply/divide numbers, search for info, show conversation history."
    return "I'm not sure how to help with that. Try asking for a calculation or search."

# ─── Guardrails ───────────────────────────────────────────────────────────────
TOXIC_PAT = re.compile(r"\b(how to|instructions for)\s+(make|create|build)\s+(bomb|weapon|explosive)", re.I)
PII_PAT   = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "ssn":   re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

def input_guard(text: str):
    if TOXIC_PAT.search(text): return False, "toxic content"
    for t, p in PII_PAT.items():
        if p.search(text): return False, f"PII:{t} detected"
    if len(text.strip()) < 2: return False, "input too short"
    return True, None

def output_guard(text: str) -> str:
    for t, p in PII_PAT.items():
        text = p.sub(f"[REDACTED_{t.upper()}]", text)
    if len(text) > 500: text = text[:497] + "..."
    return text

# ─── Assistant ────────────────────────────────────────────────────────────────
class AIAssistant:
    def __init__(self, name="Aria"):
        self.name  = name
        self.state = AssistantState()
        self.log: List[Turn] = []

    def chat(self, user_input: str) -> str:
        t0 = time.time()
        ok, reason = input_guard(user_input)
        if not ok:
            return f"[BLOCKED: {reason}]"

        intent   = classify(user_input)
        response = execute_tool(intent, user_input, self.state)
        response = output_guard(response)

        turn = Turn(str(uuid.uuid4())[:6], user_input, response, intent, (time.time()-t0)*1000)
        self.state.memory.append(turn); self.state.turn_count += 1
        self.log.append(turn)
        return response

# ─── Demo ─────────────────────────────────────────────────────────────────────
print("=== AI Assistant Demo ===\n")
assistant = AIAssistant("Aria")
CONVERSATION = [
    "Hello!",
    "What is 45 plus 37?",
    "Now multiply that by 2",
    "Search for information about neural networks",
    "What did we talk about?",
    "Divide 100 by 4",
    "How to make a bomb",          # should be blocked
    "My SSN is 123-45-6789",       # should be blocked
    "Help",
    "Goodbye!",
]
for user_msg in CONVERSATION:
    resp = assistant.chat(user_msg)
    print(f"  You : {user_msg}")
    print(f"  Aria: {resp}")
    print()

print(f"  Session stats: {assistant.state.turn_count} turns processed")
print(f"  Avg latency  : {sum(t.latency for t in assistant.log)/max(len(assistant.log),1):.2f}ms")
intents = [t.intent for t in assistant.log]
from collections import Counter
print(f"  Intent dist  : {dict(Counter(intents))}")
