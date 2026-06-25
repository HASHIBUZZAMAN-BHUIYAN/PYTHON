# Advanced Day 39 — Multi-Agent Systems
# ~10 MB RAM, <1s on CPU

print("""
=== Multi-Agent Systems — Day 39 ===

Multiple specialized agents can collaborate to solve tasks no single agent
handles well. Key patterns:

  1. Pipeline     — A → B → C (output flows through stages)
  2. Debate       — Proposer vs Critic; Judge decides
  3. Parallel     — all agents work independently; results merged
  4. Negotiation  — agents converge on a shared decision
  5. Ensemble     — aggregate answers (majority vote, average)
""")

from dataclasses import dataclass, field
from typing import List, Callable, Optional

# ─── MESSAGE PROTOCOL ────────────────────────────────────────────────────────
@dataclass
class Message:
    sender:   str
    receiver: str
    content:  str
    msg_type: str = "text"
    round:    int = 0

# ─── BASE AGENT ──────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, name, role, llm_fn=None):
        self.name   = name
        self.role   = role
        self.llm_fn = llm_fn or (lambda p: f"[{name}]: {p[:60]}")
        self.inbox: List[Message] = []

    def send(self, receiver, content, msg_type="text", rnd=0):
        return Message(self.name, receiver, content, msg_type, rnd)

    def respond(self, context=""):
        return self.llm_fn(context)

# ─── 1. PIPELINE ORCHESTRATOR ────────────────────────────────────────────────
print("=== 1. Pipeline Orchestration ===")

class PipelineOrchestrator:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.log: List[Message] = []

    def run(self, initial_input: str):
        current = initial_input
        for agent in self.agents:
            msg    = Message("user", agent.name, current)
            result = agent.respond(current)
            out_msg= agent.send("next", result)
            self.log.append(msg); self.log.append(out_msg)
            print(f"  [{agent.name}({agent.role})]: {result[:70]}")
            current = result
        return current

def researcher_fn(p): return f"Researched topic: '{p[:40]}'. Found: ML has 3 paradigms."
def writer_fn(p):     return f"Article draft: {p[:50]}... Readable summary prepared."
def reviewer_fn(p):   return f"Reviewed: '{p[:40]}'. Approved. Quality score: 8/10."

pipeline = PipelineOrchestrator([
    Agent("Alice", "Researcher", researcher_fn),
    Agent("Bob",   "Writer",     writer_fn),
    Agent("Carol", "Reviewer",   reviewer_fn),
])
result = pipeline.run("Write an article about machine learning")
print(f"\n  Final output: {result[:80]}\n")

# ─── 2. DEBATE ORCHESTRATOR ──────────────────────────────────────────────────
print("=== 2. Debate Orchestration ===")

class DebateOrchestrator:
    def __init__(self, proposer, critic, judge, rounds=2):
        self.proposer = proposer; self.critic = critic
        self.judge = judge; self.rounds = rounds
        self.log = []

    def run(self, topic):
        print(f"  Debate topic: '{topic}'")
        prop_pos = self.proposer.respond(f"Argue FOR: {topic}")
        print(f"  [{self.proposer.name}] {prop_pos[:70]}")
        self.log.append(("PROPOSE", prop_pos))

        for r in range(1, self.rounds+1):
            critique = self.critic.respond(f"Challenge this position: {prop_pos[:60]}")
            print(f"  [{self.critic.name}] R{r}: {critique[:70]}")
            self.log.append(("CRITIQUE", critique))
            prop_pos = self.proposer.respond(f"Defend against: {critique[:50]}")
            print(f"  [{self.proposer.name}] R{r} rebuttal: {prop_pos[:70]}")
            self.log.append(("REBUT", prop_pos))

        judgment = self.judge.respond(f"Topic: {topic}. Decide: {prop_pos[:60]}")
        print(f"  [{self.judge.name}] VERDICT: {judgment[:80]}")
        return judgment

def prop_fn(p): return f"FOR: '{p[:30]}' because efficiency improves outcomes."
def crit_fn(p): return f"AGAINST: '{p[:30]}' — risks include bias and opacity."
def judge_fn(p): return f"VERDICT: Both arguments valid. Recommend: measured adoption with oversight."

debate = DebateOrchestrator(
    Agent("Dan",   "Proposer", prop_fn),
    Agent("Eve",   "Critic",   crit_fn),
    Agent("Frank", "Judge",    judge_fn),
    rounds=2
)
verdict = debate.run("AI should replace human decision-making in hiring")
print()

# ─── 3. PARALLEL ORCHESTRATOR ────────────────────────────────────────────────
print("=== 3. Parallel + Majority Vote ===")

def expert_a(p): return "POSITIVE" if "good" in p.lower() or "great" in p.lower() else "NEGATIVE"
def expert_b(p): return "POSITIVE" if p.count("!") >= 1 else "NEGATIVE"
def expert_c(p): words=p.lower().split(); return "POSITIVE" if sum(1 for w in words if len(w)>5)>2 else "NEGATIVE"

agents = [Agent("ExpA","Lexicon",expert_a), Agent("ExpB","Punct",expert_b), Agent("ExpC","Length",expert_c)]
query  = "This is a great product! I absolutely love it."
votes  = [a.respond(query) for a in agents]
from collections import Counter
majority = Counter(votes).most_common(1)[0][0]
print(f"  Query: '{query}'")
print(f"  Votes: {[f'{a.name}={v}' for a,v in zip(agents,votes)]}")
print(f"  Majority decision: {majority}")

print("""
=== Summary ===
  Pipeline    — sequential processing, each agent builds on the last
  Debate      — adversarial argument for better decisions
  Parallel    — diverse perspectives, vote or average for robustness
  Negotiation — converging offers for collaborative decisions
""")
