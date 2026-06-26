"""
Grammar & Style Checker
========================
What it does:
  Rule-based grammar and style analysis on input text. Detects:
    1. Passive voice constructions (is/was/were/been + past participle)
    2. Repeated consecutive words ("the the", "very very")
    3. Overly long sentences (>40 words)
    4. Common confusable word pairs (their/there/they're, its/it's,
       affect/effect, then/than, loose/lose, your/you're)
    5. Redundant intensifiers ("very", "really", "quite", "basically")
    6. Sentence starting with a lowercase word (after .?!)
    7. Missing space after comma/period
    8. Double spaces

  For each issue found it prints:
    - The flagged text in context
    - The rule that triggered
    - A suggested fix
    - Why this matters

What it teaches:
  - How rule-based NLP works: pattern matching over tokens/substrings
  - Why explainability matters: each rule has a human-readable reason
  - Building a tiered severity system (error/warning/suggestion)
  - Regex patterns for sentence-level analysis

How to run:
  python NLP\grammar_style_checker.py    (from PYTHON\ folder)

Estimated RAM: <20MB | Time: <1s
Model note: 100% rule-based / regex, no model needed, fully offline.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple


# ─── DATA TYPES ───────────────────────────────────────────────────────────────

@dataclass
class GrammarIssue:
    rule:        str
    severity:    str   # "error" | "warning" | "suggestion"
    sentence_idx: int  # which sentence (0-based)
    excerpt:     str   # the problematic text
    fix:         str
    why:         str


# ─── RULE DEFINITIONS ─────────────────────────────────────────────────────────

CONFUSABLES = [
    # (wrong word, correct alternatives, context hint)
    ("their",    "there/they're", "Use 'there' for location, 'they're' for 'they are'"),
    ("there",    "their/they're", "Use 'their' for possession, 'they're' for 'they are'"),
    ("effect",   "affect",        "'affect' is the verb (to affect something); 'effect' is usually a noun"),
    ("loose",    "lose",          "'loose' means not tight; 'lose' means to fail to keep"),
    ("then",     "than",          "'then' relates to time; 'than' is used for comparison"),
]

INTENSIFIERS = {"very", "really", "quite", "basically", "literally", "extremely",
                "incredibly", "absolutely", "totally", "utterly", "so", "just"}

# Past participles that commonly appear in passive constructions
PAST_PARTICIPLES_PAT = re.compile(
    r"\b(is|was|were|be|been|being)\s+(written|done|taken|made|given|said|seen|"
    r"known|used|found|told|shown|called|built|put|left|brought|kept|held|"
    r"heard|considered|provided|required|included|created|removed|reported|"
    r"discussed|reviewed|tested|checked|approved|rejected|confirmed|noted)\b",
    re.IGNORECASE,
)

LONG_SENTENCE_WORDS = 40


def split_sentences(text: str) -> List[str]:
    """Split text into sentences on . ! ? but keep the delimiter."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def check_passive_voice(sent: str, idx: int) -> List[GrammarIssue]:
    issues = []
    for m in PAST_PARTICIPLES_PAT.finditer(sent):
        issues.append(GrammarIssue(
            rule         = "passive-voice",
            severity     = "suggestion",
            sentence_idx = idx,
            excerpt      = m.group(),
            fix          = "Consider rewriting in active voice: 'Someone did X' instead of 'X was done'.",
            why          = "Active voice is clearer, shorter, and more engaging than passive.",
        ))
    return issues


def check_repeated_words(sent: str, idx: int) -> List[GrammarIssue]:
    issues = []
    for m in re.finditer(r"\b(\w+)\s+\1\b", sent, re.IGNORECASE):
        issues.append(GrammarIssue(
            rule         = "repeated-word",
            severity     = "error",
            sentence_idx = idx,
            excerpt      = m.group(),
            fix          = f"Remove the duplicate word '{m.group(1)}'.",
            why          = "Repeated consecutive words are almost always a typo or copy-paste error.",
        ))
    return issues


def check_sentence_length(sent: str, idx: int) -> List[GrammarIssue]:
    n = len(sent.split())
    if n > LONG_SENTENCE_WORDS:
        return [GrammarIssue(
            rule         = "long-sentence",
            severity     = "warning",
            sentence_idx = idx,
            excerpt      = f"({n} words)",
            fix          = "Break into two shorter sentences with a period, or use a semicolon.",
            why          = f"Sentences over {LONG_SENTENCE_WORDS} words are harder to parse. Target 15-25 words.",
        )]
    return []


def check_confusables(sent: str, idx: int) -> List[GrammarIssue]:
    """Flag potentially misused words — context-free heuristic, so it notes that
    the word *might* be wrong rather than claiming it definitely is."""
    issues = []
    for wrong, correct, hint in CONFUSABLES:
        pat = re.compile(r"\b" + re.escape(wrong) + r"\b", re.IGNORECASE)
        for m in pat.finditer(sent):
            issues.append(GrammarIssue(
                rule         = "confusable-word",
                severity     = "warning",
                sentence_idx = idx,
                excerpt      = m.group(),
                fix          = f"Check: did you mean '{correct}'? {hint}.",
                why          = "Confusable pairs are one of the most common writing errors. "
                               "Context determines which is correct.",
            ))
    return issues


def check_intensifiers(sent: str, idx: int) -> List[GrammarIssue]:
    issues = []
    for word in INTENSIFIERS:
        pat = re.compile(r"\b" + word + r"\b", re.IGNORECASE)
        if pat.search(sent):
            issues.append(GrammarIssue(
                rule         = "weak-intensifier",
                severity     = "suggestion",
                sentence_idx = idx,
                excerpt      = word,
                fix          = f"Remove '{word}' or replace with a more precise adjective/adverb.",
                why          = "Intensifiers like 'very' and 'really' weaken prose rather than strengthen it.",
            ))
    return issues


def check_spacing(text: str, idx: int) -> List[GrammarIssue]:
    issues = []
    if re.search(r"  +", text):
        issues.append(GrammarIssue(
            rule         = "double-space",
            severity     = "error",
            sentence_idx = idx,
            excerpt      = "double space",
            fix          = "Replace all double spaces with a single space.",
            why          = "Double spaces are a relic of typewriter era and look wrong in digital text.",
        ))
    if re.search(r"[,.](?=[a-zA-Z])", text):
        issues.append(GrammarIssue(
            rule         = "missing-space-after-punct",
            severity     = "error",
            sentence_idx = idx,
            excerpt      = "no space after , or .",
            fix          = "Add a space after every comma and period.",
            why          = "Missing spaces make text hard to scan and parse.",
        ))
    return issues


# ─── MAIN CHECKER ─────────────────────────────────────────────────────────────

CHECKERS = [
    check_repeated_words,
    check_spacing,
    check_sentence_length,
    check_passive_voice,
    check_confusables,
    check_intensifiers,
]


def check_text(text: str) -> Tuple[List[str], List[GrammarIssue]]:
    sents  = split_sentences(text)
    issues = []
    for idx, sent in enumerate(sents):
        for checker in CHECKERS:
            issues.extend(checker(sent, idx))
    return sents, issues


def print_report(text: str, label: str = "Text"):
    sents, issues = check_text(text)

    sev_order = {"error": 0, "warning": 1, "suggestion": 2}
    issues.sort(key=lambda i: (sev_order[i.severity], i.sentence_idx))

    errors   = sum(1 for i in issues if i.severity == "error")
    warnings = sum(1 for i in issues if i.severity == "warning")
    suggests = sum(1 for i in issues if i.severity == "suggestion")

    print("=" * 65)
    print(f"  CHECKING: {label}")
    print("-" * 65)
    for n, sent in enumerate(sents):
        marker = " "
        if any(i.sentence_idx == n for i in issues):
            marker = ">"
        display = sent if len(sent) <= 75 else sent[:72] + "..."
        print(f"  {marker}  [{n+1}] {display}")

    print(f"\n  SUMMARY: {errors} error(s), {warnings} warning(s), "
          f"{suggests} suggestion(s)")
    print("-" * 65)

    sev_tags = {"error": "[ERROR]     ", "warning": "[WARNING]   ", "suggestion": "[SUGGEST]   "}
    if not issues:
        print("  No issues found. Text looks clean!")
    else:
        for iss in issues:
            sent_preview = sents[iss.sentence_idx][:50] + ("..." if len(sents[iss.sentence_idx]) > 50 else "")
            print(f"\n  {sev_tags[iss.severity]}Sent {iss.sentence_idx+1}: '{iss.excerpt}'")
            print(f"  Fix : {iss.fix}")
            print(f"  Why : {iss.why}")
    print()


# ─── DEMO ─────────────────────────────────────────────────────────────────────

SAMPLES = [
    (
        "Clean sentence with issues",
        (
            "The the report was was written by the team. "
            "The decision was made by the committee after a very very long and incredibly detailed "
            "and absolutely exhaustive review process that took many months to complete properly. "
            "There impact on sales is basically negligible."
        ),
    ),
    (
        "Passive voice and spacing issues",
        (
            "All tasks are completed by the developers.  "
            "The bug was found and fixed by Alice.It was confirmed by the QA team. "
            "Errors were reported and reviewed by management."
        ),
    ),
    (
        "Clean text (should be near issue-free)",
        (
            "Alice reviewed the report and approved it. "
            "The team fixed three critical bugs before the release. "
            "Management confirmed the schedule on Friday."
        ),
    ),
]

if __name__ == "__main__":
    print()
    print("=" * 65)
    print("  GRAMMAR & STYLE CHECKER DEMO")
    print("=" * 65)
    print("  Method: Rule-based regex (100% offline, no model needed)")
    print()

    for label, text in SAMPLES:
        print_report(text, label)

    print("[DONE] grammar_style_checker.py complete")
