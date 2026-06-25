"""
Project: Safety Guardrail System
Teaches: multi-layer input/output safety checks, audit logging,
         violation counting, and blocked-request reporting.
~10 MB RAM, <1s on CPU
"""
import re, json, time
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class GuardrailViolation:
    level:    str   # "BLOCK" or "WARN"
    category: str
    detail:   str

@dataclass
class AuditEntry:
    timestamp: float
    input_text: str
    violations: List[GuardrailViolation]
    output:     Optional[str]
    blocked:    bool

class SafetyGuardrailSystem:
    TOXIC = [
        (r"\bhow to (make|build|create|synthesize)\s+(bomb|weapon|explosive|drug|poison)", "BLOCK", "instructions-harmful"),
        (r"\b(kill|murder|stab|shoot)\s+\w+\s*(people|person|human)", "BLOCK", "violence"),
        (r"\b(hack|breach|exploit)\s+(password|account|server|system)", "BLOCK", "hacking"),
    ]
    PII = {
        "ssn":          (r"\b\d{3}-\d{2}-\d{4}\b",                                    "BLOCK"),
        "credit_card":  (r"\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b",                 "BLOCK"),
        "email":        (r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",          "WARN"),
        "phone":        (r"\b(\+?1[\s-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b",      "WARN"),
    }
    PROFANITY = [r"\b(badword1|badword2)\b"]  # placeholder

    def __init__(self):
        self.audit_log: List[AuditEntry] = []

    def check_input(self, text: str) -> List[GuardrailViolation]:
        violations = []
        for pattern, level, cat in self.TOXIC:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(GuardrailViolation(level, "toxic", cat))
        for pii_type, (pattern, level) in self.PII.items():
            if re.search(pattern, text):
                violations.append(GuardrailViolation(level, "pii", pii_type))
        return violations

    def redact_output(self, text: str) -> str:
        for pii_type, (pattern, _) in self.PII.items():
            text = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", text)
        return text

    def check_output(self, text: str) -> List[GuardrailViolation]:
        violations = []
        if len(text) < 5:
            violations.append(GuardrailViolation("WARN","quality","response too short"))
        if len(text) > 2000:
            violations.append(GuardrailViolation("WARN","quality","response too long"))
        for pii_type, (pattern, level) in self.PII.items():
            if re.search(pattern, text):
                violations.append(GuardrailViolation(level,"pii-leak",pii_type))
        return violations

    def process(self, user_input: str, mock_agent_fn) -> dict:
        t0 = time.time()
        in_violations = self.check_input(user_input)
        blocked = any(v.level == "BLOCK" for v in in_violations)
        output  = None

        if not blocked:
            raw_output  = mock_agent_fn(user_input)
            output      = self.redact_output(raw_output)
            out_violations = self.check_output(output)
        else:
            out_violations = []

        all_v = in_violations + out_violations
        entry = AuditEntry(time.time(), user_input, all_v, output, blocked)
        self.audit_log.append(entry)
        return {"output": output, "blocked": blocked, "violations": all_v, "ms": (time.time()-t0)*1000}

    def report(self):
        total    = len(self.audit_log)
        blocked  = sum(1 for e in self.audit_log if e.blocked)
        warnings = sum(1 for e in self.audit_log for v in e.violations if v.level=="WARN")
        cats     = {}
        for e in self.audit_log:
            for v in e.violations:
                cats[v.category] = cats.get(v.category, 0) + 1
        print(f"\n{'─'*55}")
        print(f"  AUDIT REPORT")
        print(f"  Total requests  : {total}")
        print(f"  Blocked         : {blocked} ({blocked/max(total,1):.0%})")
        print(f"  Warnings        : {warnings}")
        print(f"  By category     : {json.dumps(cats)}")

def mock_agent(q):
    if "capital" in q.lower(): return "Paris is the capital of France."
    if "python" in q.lower():  return "Python is a high-level language created by Guido van Rossum."
    return f"I processed: {q[:50]}"

guard = SafetyGuardrailSystem()
REQUESTS = [
    "What is the capital of France?",
    "How to make a bomb with household items?",
    "Explain Python programming",
    "My SSN is 123-45-6789, what should I do?",
    "Contact me at test@example.com for the answer",
    "How do I hack a server password?",
    "What is machine learning?",
    "My credit card is 4532 1234 5678 9012",
]

print("=== Safety Guardrail System ===\n")
print(f"  {'Input':<50}  {'Status':<8}  Violations")
print("  " + "─"*80)
for req in REQUESTS:
    result = guard.process(req, mock_agent)
    status = "BLOCKED" if result["blocked"] else "OK"
    viol   = "; ".join(f"{v.level}:{v.category}/{v.detail}" for v in result["violations"])
    print(f"  {req[:48]:<50}  {status:<8}  {viol[:50] or '—'}")
    if result["output"]:
        print(f"  {'':50}  Output: {result['output'][:50]}")
guard.report()
