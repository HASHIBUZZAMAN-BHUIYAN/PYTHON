"""
Negotiation Agent
==================
What it does:
  Simulates a goal-directed price negotiation between a Buyer agent and
  a Seller agent over multiple rounds. Each agent has:
    - An opening offer
    - A target price (what they'd ideally settle at)
    - A walk-away limit (they will not go beyond this price)
    - A strategy that controls how quickly they concede

  Strategies:
    "linear"      -- concede a fixed amount each round
    "exponential" -- concede fast early, slow later (Boulware)
    "cooperative" -- concede faster to reach agreement sooner

  The negotiation ends when:
    (a) the offers overlap (buyer price >= seller price) -> DEAL
    (b) either agent reaches their walk-away limit without agreement -> NO DEAL
    (c) max rounds exceeded -> NO DEAL

  A full transcript is printed each round, then a final outcome.
  Three scenarios are demoed to show different negotiation dynamics.

What it teaches:
  - Autonomous goal-directed agents with constraints
  - Concession strategies and their tradeoffs
  - Walk-away threshold as a hard constraint
  - How strategy choice affects outcome (speed, final price)

How to run:
  python negotiation_agent.py

API key needed? NO -- fully offline. Pure Python math, no API key.
"""

import math
from dataclasses import dataclass
from typing import Optional


# ─── CONCESSION STRATEGIES ────────────────────────────────────────────────────

def linear_concession(opening: float, target: float, limit: float,
                      round_num: int, max_rounds: int) -> float:
    """Concede equal steps each round until target, then limit."""
    progress = min(1.0, round_num / max_rounds)
    return opening + (target - opening) * progress

def exponential_concession(opening: float, target: float, limit: float,
                           round_num: int, max_rounds: int) -> float:
    """Boulware-ish: concede fast early, then slow down near target."""
    progress = 1 - math.exp(-3 * round_num / max_rounds)
    return opening + (target - opening) * progress

def cooperative_concession(opening: float, target: float, limit: float,
                           round_num: int, max_rounds: int) -> float:
    """Concede quickly toward midpoint of target+limit to close deal fast."""
    midpoint = (target + limit) / 2
    progress = min(1.0, round_num / (max_rounds * 0.6))  # reaches mid by 60% of rounds
    return opening + (midpoint - opening) * progress

STRATEGIES = {
    "linear":      linear_concession,
    "exponential": exponential_concession,
    "cooperative": cooperative_concession,
}


# ─── AGENT ────────────────────────────────────────────────────────────────────

@dataclass
class NegotiationAgent:
    name:      str
    role:      str          # "buyer" | "seller"
    opening:   float        # first offer
    target:    float        # ideal settlement
    limit:     float        # walk-away price
    strategy:  str          # "linear" | "exponential" | "cooperative"

    def offer(self, round_num: int, max_rounds: int) -> float:
        """Compute this agent's offer for a given round."""
        fn = STRATEGIES[self.strategy]
        raw = fn(self.opening, self.target, self.limit, round_num, max_rounds)
        # Buyer: never exceed limit. Seller: never go below limit.
        if self.role == "buyer":
            return min(raw, self.limit)
        else:
            return max(raw, self.limit)

    def at_limit(self, current_offer: float) -> bool:
        if self.role == "buyer":
            return current_offer >= self.limit
        return current_offer <= self.limit


# ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────

@dataclass
class NegotiationResult:
    deal_reached:  bool
    final_price:   Optional[float]
    rounds_taken:  int
    reason:        str


class NegotiationOrchestrator:
    """
    Runs the negotiation loop:
      Round 0: buyer makes opening bid, seller makes opening ask
      Round N: each agent updates offer based on strategy + round number
      Check: if buyer_offer >= seller_offer -> DEAL at midpoint
             if either hits walk-away      -> NO DEAL
    """

    def __init__(self, buyer: NegotiationAgent, seller: NegotiationAgent,
                 max_rounds: int = 10, item: str = "the item"):
        self.buyer      = buyer
        self.seller     = seller
        self.max_rounds = max_rounds
        self.item       = item

    def run(self, verbose: bool = True) -> NegotiationResult:

        def sep(char="-", width=64): return "  " + char * width

        if verbose:
            print(sep("="))
            print(f"  NEGOTIATION: {self.item}")
            print(sep("="))
            print(f"  {self.buyer.name:<20} wants to PAY  <= ${self.buyer.limit:,.0f}  "
                  f"(opens ${self.buyer.opening:,.0f}, target ${self.buyer.target:,.0f})")
            print(f"  {self.seller.name:<20} wants to GET >= ${self.seller.limit:,.0f}  "
                  f"(opens ${self.seller.opening:,.0f}, target ${self.seller.target:,.0f})")
            print(sep())
            print(f"  {'Round':<7}  {'Buyer Bid':>12}  {'Seller Ask':>12}  "
                  f"{'Gap':>10}  Status")
            print(sep())

        buyer_offer  = self.buyer.opening
        seller_offer = self.seller.opening

        for rnd in range(0, self.max_rounds + 1):
            buyer_offer  = self.buyer.offer(rnd, self.max_rounds)
            seller_offer = self.seller.offer(rnd, self.max_rounds)
            gap          = seller_offer - buyer_offer

            if buyer_offer >= seller_offer:
                deal_price = (buyer_offer + seller_offer) / 2
                if verbose:
                    print(f"  {rnd:<7}  ${buyer_offer:>10,.0f}  ${seller_offer:>10,.0f}  "
                          f"{'GAP CLOSED':>10}  -> DEAL at ${deal_price:,.0f}")
                    print(sep())
                    print(f"  OUTCOME: DEAL at ${deal_price:,.0f} after {rnd} round(s)")
                    print(f"  Buyer paid ${deal_price - self.buyer.opening:+,.0f} vs opening. "
                          f"Seller got ${deal_price - self.seller.opening:+,.0f} vs opening.")
                return NegotiationResult(True, deal_price, rnd, "Offers converged")

            buyer_stuck  = self.buyer.at_limit(buyer_offer)
            seller_stuck = self.seller.at_limit(seller_offer)

            status = ""
            if rnd == self.max_rounds:
                status = "MAX ROUNDS"
            elif buyer_stuck and seller_stuck:
                status = "BOTH AT LIMIT"
            elif buyer_stuck:
                status = "BUYER AT LIMIT"
            elif seller_stuck:
                status = "SELLER AT LIMIT"

            if verbose:
                print(f"  {rnd:<7}  ${buyer_offer:>10,.0f}  ${seller_offer:>10,.0f}  "
                      f"${gap:>9,.0f}  {status}")

            if status:
                if verbose:
                    print(sep())
                    print(f"  OUTCOME: NO DEAL  ({status})  Gap remaining: ${gap:,.0f}")
                return NegotiationResult(False, None, rnd, status)

        # Fallback (shouldn't reach here normally)
        return NegotiationResult(False, None, self.max_rounds, "Max rounds exceeded")


# ─── DEMO ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    orch = NegotiationOrchestrator

    print()
    print("=" * 65)
    print("  NEGOTIATION AGENT DEMO")
    print("=" * 65)

    # ── Scenario 1: Car purchase — linear strategies, deal reached at round 8 ─
    # ZOPA: buyer limit $25k vs seller limit $22k -> overlap exists.
    # Both targets pass through the overlap on the last round.
    print()
    r1 = orch(
        buyer  = NegotiationAgent("Alice (Buyer)",  "buyer",  18000, 24000, 25000, "linear"),
        seller = NegotiationAgent("Bob (Seller)",   "seller", 28000, 23000, 22000, "linear"),
        max_rounds = 8,
        item   = "Used Car (2019 Honda Civic)"
    ).run()

    # ── Scenario 2: Freelance contract — cooperative buyer reaches seller fast ─
    # Cooperative buyer pushes to $5,750 by round 6; exponential seller
    # softens to $5,727 by round 8 -> buyer > seller -> DEAL.
    print()
    r2 = orch(
        buyer  = NegotiationAgent("Startup (Buyer)",   "buyer",  3000, 5500, 6000,  "cooperative"),
        seller = NegotiationAgent("Dev (Seller)",      "seller", 8000, 5500, 5000,  "exponential"),
        max_rounds = 10,
        item   = "Freelance Dev Contract ($/month)"
    ).run()

    # ── Scenario 3: Real estate — ZOPA doesn't exist, NO DEAL ────────────────
    # Buyer's hard limit $340k < seller's hard limit $370k -> no overlap possible.
    print()
    r3 = orch(
        buyer  = NegotiationAgent("Chen Family",   "buyer",  300000, 330000, 340000, "linear"),
        seller = NegotiationAgent("Estate Agent",  "seller", 420000, 390000, 370000, "exponential"),
        max_rounds = 8,
        item   = "3-Bed House (fictional listing)"
    ).run()

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  SCENARIO SUMMARY")
    print("  " + "-" * 61)
    for label, result in [
        ("Car purchase",        r1),
        ("Freelance contract",  r2),
        ("Real estate",         r3),
    ]:
        outcome = f"DEAL at ${result.final_price:,.0f}" if result.deal_reached else f"NO DEAL ({result.reason})"
        print(f"  {label:<22}: {outcome}  ({result.rounds_taken} rounds)")
