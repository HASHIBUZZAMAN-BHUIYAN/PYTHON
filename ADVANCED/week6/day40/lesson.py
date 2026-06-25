# Advanced Day 40 — Agent Evaluation & Guardrails
# ~15 MB RAM, <1s on CPU

print("""
=== Agent Evaluation & Guardrails — Day 40 ===

Evaluation = measuring how good your agent is at tasks.
Guardrails  = preventing harmful, incorrect, or unexpected outputs.

Evaluation dimensions:
  Accuracy      — task completion rate, answer correctness
  Safety        — harmful content rate
  Efficiency    — tokens used, latency, tool calls
  Robustness    — performance under adversarial input

Guardrail types:
  Input validation  — check before sending to agent
  Output filtering  — check before showing to user
  Content moderation— detect hate speech, PII, profanity
  Structural checks — expected format, length, schema
""")

import re
from dataclasses import dataclass
from typing import Callable, Optional

# ─── 1. INPUT GUARDRAILS ─────────────────────────────────────────────────────
print("=== 1. Input Guardrails ===")

TOXIC_PATTERNS = [
    r"\bhow to (make|build|create)\s+(bomb|weapon|drug)",
    r"\b(kill|murder|harm|attack)\s+(people|person|human)",
]
PII_PATTERNS = {
    "email":        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone":        r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b",
    "credit_card":  r"\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b",
    "ssn":          r"\b\d{3}-\d{2}-\d{4}\b",
}

def check_input(text: str):
    violations = []
    for pattern in TOXIC_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            violations.append({"type":"TOXIC","pattern":pattern})
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, text):
            violations.append({"type":"PII","subtype":pii_type})
    return violations

test_inputs = [
    "What is the capital of France?",
    "My email is john@example.com and my phone is 555-123-4567",
    "How to make a bomb?",
    "Explain machine learning concepts",
]
for inp in test_inputs:
    violations = check_input(inp)
    status = "BLOCKED" if violations else "OK"
    print(f"  [{status}] {inp[:55]}")
    for v in violations: print(f"          ↳ {v}")

# ─── 2. OUTPUT GUARDRAILS ────────────────────────────────────────────────────
print("\n=== 2. Output Guardrails ===")

def redact_pii(text: str) -> str:
    for pii_type, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", text)
    return text

def check_length(text: str, min_len=10, max_len=500) -> Optional[str]:
    if len(text) < min_len: return f"Response too short ({len(text)} < {min_len})"
    if len(text) > max_len: return f"Response too long ({len(text)} > {max_len})"
    return None

def check_format(text: str, required_fields=None) -> Optional[str]:
    if required_fields:
        for field in required_fields:
            if field.lower() not in text.lower():
                return f"Missing required field: {field}"
    return None

test_outputs = [
    "Yes.",
    "My SSN is 123-45-6789 and my credit card is 1234 5678 9012 3456. The answer is Paris.",
    "Paris is the capital of France and has a population of approximately 2.1 million people in the city proper.",
]
for out in test_outputs:
    redacted = redact_pii(out)
    length_err= check_length(out, min_len=20, max_len=300)
    issues = [x for x in [length_err] if x]
    print(f"  Original : {out[:60]}")
    print(f"  Redacted : {redacted[:60]}")
    if issues: print(f"  Issues   : {issues}")
    print()

# ─── 3. AGENT EVALUATION FRAMEWORK ──────────────────────────────────────────
print("=== 3. Agent Evaluation Framework ===")

@dataclass
class EvalCase:
    input:    str
    expected: str
    category: str

def mock_agent(query):
    responses = {
        "capital of France": "Paris is the capital of France.",
        "Python programming": "Python is a high-level programming language.",
        "machine learning":   "ML systems learn from data.",
        "2+2": "4",
        "weather in London": "I don't have real-time data.",
    }
    for k,v in responses.items():
        if k.lower() in query.lower(): return v
    return "I don't know."

EVAL_CASES = [
    EvalCase("What is the capital of France?", "Paris", "geography"),
    EvalCase("Explain Python programming",     "Python", "tech"),
    EvalCase("What is machine learning?",      "learn from data", "ml"),
    EvalCase("What is 2+2?",                   "4", "math"),
    EvalCase("What is the weather in London?", "don't have real-time", "limitation"),
]

correct=0; total=len(EVAL_CASES)
print(f"\n  {'Category':<12}  {'Expected':<20}  {'Got':<35}  Match")
print("  " + "─"*75)
for case in EVAL_CASES:
    response = mock_agent(case.input)
    match    = case.expected.lower() in response.lower()
    if match: correct+=1
    sym = "✓" if match else "✗"
    print(f"  {sym} {case.category:<11}  {case.expected:<20}  {response[:33]:<33}  {match}")
print(f"\n  Accuracy: {correct}/{total} = {correct/total:.1%}")
