"""
Project: Policy Debate System
Teaches: adversarial multi-agent debate with structured arguments,
         judge scoring, and debate transcript generation.
~10 MB RAM, <1s on CPU
"""
import random

class DebateAgent:
    def __init__(self, name, stance, arguments):
        self.name      = name
        self.stance    = stance
        self.arguments = arguments
        self.rebuttals = []

    def open_argument(self):
        return f"[{self.stance.upper()}] {self.arguments[0]}"

    def rebuttal(self, opponent_arg, round_num):
        idx = min(round_num, len(self.arguments)-1)
        return f"[{self.stance.upper()} R{round_num}] Regarding '{opponent_arg[:30]}...' — {self.arguments[idx]}"

class JudgeAgent:
    CRITERIA = ["clarity", "evidence", "logic", "persuasiveness"]
    def __init__(self, name):
        self.name = name

    def score(self, argument, seed=None):
        if seed: random.seed(seed)
        return {c: random.randint(5, 10) for c in self.CRITERIA}

    def decide(self, for_scores, against_scores):
        for_total     = sum(sum(r.values()) for r in for_scores)
        against_total = sum(sum(r.values()) for r in against_scores)
        if for_total > against_total:
            return "FOR", for_total, against_total
        elif against_total > for_total:
            return "AGAINST", for_total, against_total
        return "TIE", for_total, against_total

# ─── Debate: Universal Basic Income ──────────────────────────────────────────
proposer = DebateAgent("Alice", "FOR", [
    "UBI ensures every citizen's basic needs are met, reducing poverty.",
    "Evidence from pilot programs in Finland and Kenya show mental health improvements.",
    "Automation is eliminating jobs; UBI provides a safety net for displaced workers.",
    "UBI simplifies the welfare bureaucracy and reduces administrative costs.",
])
critic = DebateAgent("Bob", "AGAINST", [
    "UBI is fiscally unsustainable without significant tax increases.",
    "Giving unconditional income reduces incentives to work and contribute.",
    "Resources would be better spent on targeted poverty reduction programs.",
    "Inflation would erode the value of UBI payments over time.",
])
judge = JudgeAgent("Judge Carol")

ROUNDS = 3
transcript = []
for_scores = []; against_scores = []

print("=== Policy Debate: Universal Basic Income ===\n")
print(f"{'─'*70}")
print(f"MOTION: This house believes Universal Basic Income should be implemented.")
print(f"{'─'*70}\n")

# Opening arguments
op_for     = proposer.open_argument()
op_against = critic.open_argument()
print(f"OPENING FOR:\n  {op_for}\n")
print(f"OPENING AGAINST:\n  {op_against}\n")
transcript.extend([("FOR-OPEN", op_for), ("AGAINST-OPEN", op_against)])

# Score openings
for_scores.append(judge.score(op_for, seed=0))
against_scores.append(judge.score(op_against, seed=1))

# Rebuttals
for rnd in range(1, ROUNDS+1):
    print(f"{'─'*30} Round {rnd} {'─'*30}")
    reb_for     = proposer.rebuttal(op_against, rnd)
    reb_against = critic.rebuttal(op_for, rnd)
    print(f"FOR REBUTTAL:\n  {reb_for}")
    print(f"AGAINST REBUTTAL:\n  {reb_against}\n")
    for_scores.append(judge.score(reb_for, seed=rnd*10))
    against_scores.append(judge.score(reb_against, seed=rnd*10+1))
    op_for = reb_for; op_against = reb_against

# Judge decision
winner, for_total, against_total = judge.decide(for_scores, against_scores)
print(f"{'─'*70}")
print(f"JUDGE {judge.name.upper()} VERDICT:")
print(f"  FOR total score    : {for_total}")
print(f"  AGAINST total score: {against_total}")
print(f"  WINNER: {winner}")
print(f"\n  Score breakdown per round:")
criteria = judge.CRITERIA
for i, (f, a) in enumerate(zip(for_scores, against_scores)):
    rnd_name = "OPENING" if i==0 else f"Round {i}"
    for c in criteria:
        print(f"    {rnd_name:10s}  {c:16s}  FOR={f[c]}  AGAINST={a[c]}")
