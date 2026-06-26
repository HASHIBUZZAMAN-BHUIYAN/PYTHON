# AGENTIC_AI Reference — Multi-Agent Orchestration Template
# Patterns: Pipeline, Debate, Parallel, Negotiation.
# All mock LLM — no API key required.
# ~10 MB RAM, <1s on CPU

import time
from typing import Callable

# ─── BASE AGENT ───────────────────────────────────────────────────────────────
class BaseAgent:
    def __init__(self, name: str, role: str, llm_fn: Callable = None):
        self.name    = name
        self.role    = role
        self._llm    = llm_fn or self._default_llm
        self.history = []

    def _default_llm(self, prompt: str) -> str:
        return f"[{self.name}] Response to: {prompt[:50]}"

    def think(self, prompt: str) -> str:
        response = self._llm(prompt)
        self.history.append({"prompt": prompt[:80], "response": response})
        return response

    def __repr__(self): return f"Agent({self.name}, role={self.role})"

# ─── MESSAGE ──────────────────────────────────────────────────────────────────
class Message:
    def __init__(self, sender: str, receiver: str, content: str, msg_type="text"):
        self.sender   = sender
        self.receiver = receiver
        self.content  = content
        self.msg_type = msg_type
        self.timestamp= time.time()

    def __repr__(self): return f"[{self.sender}->{self.receiver}] {self.content[:60]}"

# ─── 1. PIPELINE ORCHESTRATOR ─────────────────────────────────────────────────
class PipelineOrchestrator:
    """
    Run agents in sequence. Output of agent[i] becomes input to agent[i+1].
    """
    def __init__(self, agents: list):
        self.agents = agents
        self.log    = []

    def run(self, initial_input: str, verbose=True) -> str:
        current = initial_input
        for agent in self.agents:
            result = agent.think(current)
            msg = Message(agent.name, "pipeline", result)
            self.log.append(msg)
            if verbose:
                print(f"  [{agent.name}] -> {result[:80]}")
            current = result
        return current

# ─── 2. DEBATE ORCHESTRATOR ───────────────────────────────────────────────────
class DebateOrchestrator:
    """
    Two agents debate: proposer argues FOR, critic argues AGAINST.
    N rounds, then judge summarizes.
    """
    def __init__(self, proposer: BaseAgent, critic: BaseAgent, judge: BaseAgent = None):
        self.proposer = proposer
        self.critic   = critic
        self.judge    = judge
        self.transcript = []

    def run(self, topic: str, n_rounds: int = 3, verbose=True) -> str:
        if verbose: print(f"\n  Topic: {topic}")
        last_proposer = f"Argue FOR: {topic}"
        last_critic   = f"Counter the argument FOR: {topic}"
        for round_n in range(1, n_rounds+1):
            prop_resp  = self.proposer.think(last_proposer)
            crit_resp  = self.critic.think(last_critic)
            self.transcript.append((round_n, prop_resp, crit_resp))
            if verbose:
                print(f"\n  Round {round_n}:")
                print(f"    [PRO] {prop_resp[:80]}")
                print(f"    [CON] {crit_resp[:80]}")
            last_proposer = f"Respond to critic: {crit_resp[:60]}"
            last_critic   = f"Respond to proposer: {prop_resp[:60]}"
        conclusion = ""
        if self.judge:
            all_args = " | ".join(p[:40] for _,p,_ in self.transcript)
            conclusion = self.judge.think(f"Summarize debate on '{topic}': {all_args}")
            if verbose: print(f"\n  [JUDGE] {conclusion[:100]}")
        return conclusion

# ─── 3. PARALLEL ORCHESTRATOR ─────────────────────────────────────────────────
class ParallelOrchestrator:
    """
    All agents process the same input independently, then results are merged.
    """
    def __init__(self, agents: list, merger: BaseAgent = None):
        self.agents = agents
        self.merger = merger

    def run(self, input_text: str, verbose=True) -> list:
        results = []
        for agent in self.agents:
            r = agent.think(input_text)
            results.append((agent.name, r))
            if verbose: print(f"  [{agent.name}] {r[:80]}")
        if self.merger and results:
            merged_input = "\n".join(f"{n}: {r[:60]}" for n,r in results)
            final = self.merger.think(f"Synthesize:\n{merged_input}")
            if verbose: print(f"\n  [MERGER] {final[:100]}")
            return results, final
        return results, None

# ─── 4. NEGOTIATION ORCHESTRATOR ─────────────────────────────────────────────
class NegotiationOrchestrator:
    """
    Two agents negotiate toward an agreed value.
    buyer_fn(seller_offer) → buyer_offer
    seller_fn(buyer_offer) → seller_offer
    """
    def __init__(self, buyer: BaseAgent, seller: BaseAgent,
                 buyer_min: float, seller_min: float):
        self.buyer      = buyer
        self.seller     = seller
        self.buyer_min  = buyer_min    # buyer won't pay more
        self.seller_min = seller_min   # seller won't accept less

    def run(self, max_rounds: int = 5, verbose=True):
        buyer_offer  = self.buyer_min
        seller_offer = self.seller_min * 1.5
        if verbose: print(f"  Buyer max={self.buyer_min}  Seller min={self.seller_min}")
        for round_n in range(1, max_rounds+1):
            # Buyer moves up
            buyer_offer  = min(buyer_offer  * 1.08, self.buyer_min)
            # Seller moves down
            seller_offer = max(seller_offer * 0.92, self.seller_min)
            if verbose:
                print(f"  Round {round_n}: Buyer={buyer_offer:.1f}  Seller={seller_offer:.1f}")
            if buyer_offer >= seller_offer:
                deal = (buyer_offer + seller_offer) / 2
                if verbose: print(f"  DEAL at {deal:.1f}!")
                return True, deal
        if verbose: print("  No deal reached.")
        return False, None

# ─── DEMO ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Pipeline ===")
    def research_llm(p): return f"Research findings on '{p[:30]}': Machine learning is widely used."
    def summary_llm(p):  return f"Summary: {p[:60]}... [condensed to 2 key points]"
    def write_llm(p):    return f"Article: {p[:50]} -- This topic impacts society significantly."
    pipeline = PipelineOrchestrator([
        BaseAgent("Researcher", "research", research_llm),
        BaseAgent("Summarizer", "summarize", summary_llm),
        BaseAgent("Writer",     "write",    write_llm),
    ])
    pipeline.run("machine learning in healthcare")

    print("\n=== Debate ===")
    def pro_llm(p):   return f"FOR: AI improves productivity significantly. Humans can focus on creative work."
    def con_llm(p):   return f"CON: AI may displace jobs and reduce human autonomy. We must be cautious."
    def judge_llm(p): return f"CONCLUSION: Both sides raise valid points. Balanced regulation is recommended."
    debate = DebateOrchestrator(
        proposer=BaseAgent("Proposer", "argue for",  pro_llm),
        critic  =BaseAgent("Critic",   "argue against", con_llm),
        judge   =BaseAgent("Judge",    "summarize", judge_llm),
    )
    debate.run("AI should replace human programmers", n_rounds=2)

    print("\n=== Negotiation ===")
    neg = NegotiationOrchestrator(
        buyer  =BaseAgent("Buyer",  "buy cheap"),
        seller =BaseAgent("Seller", "sell high"),
        buyer_min=120., seller_min=90.
    )
    neg.run(max_rounds=6)
