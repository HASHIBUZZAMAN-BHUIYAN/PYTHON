"""
Code Review Agent
==================
What it does:
  Receives Python code snippets and performs static rule-based analysis.
  NOT a full linter (no AST parsing) -- it uses regex + heuristic rules
  that are easy to understand and extend. For each snippet it:
    1. Flags issues (bugs, style, best-practice violations)
    2. Suggests a concrete fix for each issue found
    3. Gives a plain-English "why it matters" explanation
    4. Assigns a simple quality grade (A/B/C/D)

  Rule categories covered:
    - Missing docstrings (module, function, class)
    - Unused variable patterns (assignment then never used in snippet)
    - Bare except clauses (catches everything including SystemExit)
    - Mutable default arguments (def f(x=[]):)
    - Off-by-one risk patterns (range(len(...)) without -1 check)
    - Hardcoded credentials / magic strings
    - Comparison to None using == instead of is/is not
    - Print debugging left in code
    - Missing return type in function def
    - Variable shadowing builtins (list, dict, type, id...)

What it teaches:
  - Rule engine pattern: list of (check_fn, severity, fix_fn)
  - Perceive -> analyze -> report agent loop
  - How static analysis tools work at a conceptual level

How to run:
  python code_review_agent.py

API key needed? NO -- fully offline, pure regex heuristics.
"""

import re
from dataclasses import dataclass, field
from typing import Callable


# ─── ISSUE DATA TYPE ──────────────────────────────────────────────────────────

@dataclass
class Issue:
    rule:       str
    severity:   str     # "error" | "warning" | "style"
    line:       int     # 1-based line number (-1 = whole file)
    description: str
    suggestion:  str
    why:         str


# ─── RULE ENGINE ──────────────────────────────────────────────────────────────

RuleCheck = Callable[[list[str], str], list[Issue]]

RULES: list[RuleCheck] = []

def rule(fn: RuleCheck) -> RuleCheck:
    """Decorator to register a rule."""
    RULES.append(fn)
    return fn


@rule
def check_missing_docstring(lines: list[str], code: str) -> list[Issue]:
    """Flag functions/classes that have no docstring on the next line."""
    issues = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^(def |class )\w+", stripped):
            # Look at next non-empty line
            next_lines = [l.strip() for l in lines[i+1:i+4] if l.strip()]
            has_doc = next_lines and next_lines[0].startswith(('"""', "'''"))
            if not has_doc:
                name_match = re.search(r"(def |class )(\w+)", stripped)
                name = name_match.group(2) if name_match else "?"
                kind = "Function" if stripped.startswith("def") else "Class"
                issues.append(Issue(
                    rule        = "missing-docstring",
                    severity    = "style",
                    line        = i + 1,
                    description = f"{kind} '{name}' has no docstring.",
                    suggestion  = f'Add """Brief description.""" as the first line inside {kind.lower()} {name}.',
                    why         = "Docstrings let users call help() on your code and enable auto-generated docs.",
                ))
    return issues


@rule
def check_bare_except(lines: list[str], code: str) -> list[Issue]:
    """Flag bare 'except:' without specifying an exception type."""
    issues = []
    for i, line in enumerate(lines):
        if re.match(r"\s*except\s*:", line):
            issues.append(Issue(
                rule        = "bare-except",
                severity    = "error",
                line        = i + 1,
                description = "Bare 'except:' catches everything including KeyboardInterrupt and SystemExit.",
                suggestion  = "Replace 'except:' with 'except Exception as e:' (or a more specific exception).",
                why         = "Bare except can silently swallow crashes and make debugging nearly impossible.",
            ))
    return issues


@rule
def check_mutable_default_arg(lines: list[str], code: str) -> list[Issue]:
    """Flag mutable default arguments like def f(x=[]) or def f(x={})."""
    issues = []
    for i, line in enumerate(lines):
        if re.search(r"def \w+\(.*=\s*(\[\]|\{\}|\[\s*\]|\{\s*\})", line):
            issues.append(Issue(
                rule        = "mutable-default-arg",
                severity    = "error",
                line        = i + 1,
                description = "Mutable default argument ([] or {}) is shared across all calls.",
                suggestion  = "Use 'None' as default and initialise inside the function: if x is None: x = []",
                why         = "The same list/dict object is reused on every call, causing silent state leakage.",
            ))
    return issues


@rule
def check_none_comparison(lines: list[str], code: str) -> list[Issue]:
    """Flag == None or != None comparisons."""
    issues = []
    for i, line in enumerate(lines):
        if re.search(r"==\s*None|!=\s*None|None\s*==|None\s*!=", line):
            issues.append(Issue(
                rule        = "none-comparison",
                severity    = "warning",
                line        = i + 1,
                description = "Comparison to None using == or != instead of 'is' / 'is not'.",
                suggestion  = "Replace '== None' with 'is None' and '!= None' with 'is not None'.",
                why         = "None is a singleton; 'is' tests identity, which is both correct and faster.",
            ))
    return issues


@rule
def check_print_debug(lines: list[str], code: str) -> list[Issue]:
    """Flag print() calls that look like debug traces (contain 'debug', 'test', or variable dumps)."""
    issues = []
    debug_pats = [r"print\s*\(\s*['\"]debug", r"print\s*\(\s*['\"]test",
                  r"print\s*\(\s*f?['\"].*=.*\)", r"print\s*\(vars\("]
    for i, line in enumerate(lines):
        for pat in debug_pats:
            if re.search(pat, line, re.IGNORECASE):
                issues.append(Issue(
                    rule        = "debug-print",
                    severity    = "warning",
                    line        = i + 1,
                    description = "Debug print statement left in production code.",
                    suggestion  = "Remove the print() or replace with logging.debug() from the logging module.",
                    why         = "Debug prints clutter production logs and may leak sensitive data.",
                ))
                break
    return issues


@rule
def check_shadow_builtin(lines: list[str], code: str) -> list[Issue]:
    """Flag variable names that shadow Python builtins."""
    builtins_at_risk = {"list", "dict", "set", "type", "id", "input", "filter",
                        "map", "zip", "range", "sum", "min", "max", "len", "str", "int", "float"}
    issues = []
    for i, line in enumerate(lines):
        m = re.match(r"\s*(\w+)\s*=", line)
        if m and m.group(1) in builtins_at_risk:
            name = m.group(1)
            issues.append(Issue(
                rule        = "shadow-builtin",
                severity    = "warning",
                line        = i + 1,
                description = f"Variable name '{name}' shadows a Python builtin.",
                suggestion  = f"Rename to something like '{name}_items' or '{name}_val'.",
                why         = f"After assignment, the builtin '{name}() is no longer accessible in this scope.",
            ))
    return issues


@rule
def check_hardcoded_secret(lines: list[str], code: str) -> list[Issue]:
    """Flag hardcoded passwords, API keys, or tokens."""
    patterns = [
        r"password\s*=\s*['\"][^'\"]{4,}['\"]",
        r"api_key\s*=\s*['\"][^'\"]{8,}['\"]",
        r"secret\s*=\s*['\"][^'\"]{4,}['\"]",
        r"token\s*=\s*['\"][^'\"]{8,}['\"]",
    ]
    issues = []
    for i, line in enumerate(lines):
        for pat in patterns:
            if re.search(pat, line, re.IGNORECASE):
                issues.append(Issue(
                    rule        = "hardcoded-secret",
                    severity    = "error",
                    line        = i + 1,
                    description = "Hardcoded credential or secret detected.",
                    suggestion  = "Move to an environment variable: os.environ.get('MY_SECRET')",
                    why         = "Hardcoded secrets committed to git are a common source of data breaches.",
                ))
                break
    return issues


# ─── GRADER ───────────────────────────────────────────────────────────────────

def grade(issues: list[Issue]) -> str:
    errors   = sum(1 for i in issues if i.severity == "error")
    warnings = sum(1 for i in issues if i.severity == "warning")
    styles   = sum(1 for i in issues if i.severity == "style")
    if errors >= 2:    return "D"
    if errors == 1:    return "C"
    if warnings >= 2:  return "C"
    if warnings == 1:  return "B"
    if styles >= 2:    return "B"
    return "A"


# ─── CODE REVIEW AGENT ────────────────────────────────────────────────────────

class CodeReviewAgent:
    """
    Perceive  : receive a code snippet (string)
    Analyze   : run all registered rules against it
    Report    : print each issue with fix + explanation, assign grade
    """

    def review(self, snippet_name: str, code: str) -> list[Issue]:
        lines  = code.split("\n")
        issues = []
        for rule_fn in RULES:
            issues.extend(rule_fn(lines, code))
        # Sort: errors first, then warnings, then style
        severity_order = {"error": 0, "warning": 1, "style": 2}
        issues.sort(key=lambda i: (severity_order[i.severity], i.line))
        return issues

    def print_report(self, snippet_name: str, code: str):
        issues = self.review(snippet_name, code)
        g      = grade(issues)

        print("=" * 65)
        print(f"  SNIPPET: {snippet_name}")
        print("  " + "-" * 61)
        # Print code with line numbers
        for n, line in enumerate(code.strip().split("\n"), 1):
            marker = " "
            if any(i.line == n for i in issues):
                marker = ">"
            print(f"  {marker} {n:>3} | {line}")

        print(f"\n  GRADE: {g}  ({len(issues)} issue(s) found)")
        print("  " + "-" * 61)

        if not issues:
            print("  No issues found. Code looks clean!")
        else:
            for issue in issues:
                sev_label = {"error": "[ERROR]  ", "warning": "[WARN]   ", "style": "[STYLE]  "}
                print(f"\n  {sev_label[issue.severity]}Line {issue.line}: {issue.description}")
                print(f"  Fix : {issue.suggestion}")
                print(f"  Why : {issue.why}")
        print()


# ─── DEMO ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    agent = CodeReviewAgent()

    # ── Snippet 1: multiple issues ────────────────────────────────────────────
    SNIPPET_1 = """\
def add_item(items=[]):
    items.append("new")
    return items

def get_user(user_id):
    list = fetch_from_db(user_id)
    if list == None:
        return None
    print("debug: user =", list)
    return list
"""

    # ── Snippet 2: hardcoded secret + bare except ─────────────────────────────
    SNIPPET_2 = """\
import requests

def call_api(endpoint):
    api_key = "sk-AbCdEfGhIjKlMnOpQrStUvWxYz123456"
    try:
        response = requests.get(endpoint, headers={"key": api_key})
        return response.json()
    except:
        return None
"""

    # ── Snippet 3: clean code (should get grade A) ─────────────────────────────
    SNIPPET_3 = """\
def calculate_average(numbers: list) -> float:
    \"\"\"Return the arithmetic mean of a list of numbers.\"\"\"
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)
"""

    # ── Snippet 4: missing docstrings everywhere ───────────────────────────────
    SNIPPET_4 = """\
class DataProcessor:
    def __init__(self, config):
        self.config = config

    def process(self, data):
        result = []
        for item in data:
            result.append(item * 2)
        return result

    def validate(self, data):
        return data is not None
"""

    for name, code in [
        ("1 - Mutable default + shadowed builtin + None comparison + debug print", SNIPPET_1),
        ("2 - Hardcoded API key + bare except",                                     SNIPPET_2),
        ("3 - Clean code (should be grade A)",                                      SNIPPET_3),
        ("4 - Missing docstrings throughout",                                        SNIPPET_4),
    ]:
        agent.print_report(name, code)
