"""
Project: Multi-Party Salary Negotiation System
Teaches: converging offer negotiation between buyer/seller agents,
         ZOPA (Zone of Possible Agreement) analysis, visualizing negotiation dynamics.
~15 MB RAM, ~1s on CPU
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random

class NegotiatorAgent:
    def __init__(self, name, initial_offer, target, limit, strategy="linear"):
        self.name    = name
        self.offer   = initial_offer
        self.target  = target
        self.limit   = limit
        self.strategy= strategy
        self.history = [initial_offer]

    def counter_offer(self, other_offer, round_num, max_rounds=10):
        # Move toward other agent's offer each round
        progress    = round_num / max_rounds
        if self.strategy == "linear":
            concession = abs(self.limit - self.target) * progress * 0.15
        elif self.strategy == "aggressive":
            concession = abs(self.limit - self.target) * progress * 0.05
        else:  # cooperative
            concession = abs(self.limit - self.target) * progress * 0.25

        if self.offer < other_offer:  # buyer moving up
            new_offer = min(self.offer + concession, self.limit)
        else:  # seller moving down
            new_offer = max(self.offer - concession, self.limit)

        self.offer = round(new_offer, 2)
        self.history.append(self.offer)
        return self.offer

class NegotiationOrchestrator:
    def __init__(self, buyer, seller, max_rounds=12):
        self.buyer      = buyer
        self.seller     = seller
        self.max_rounds = max_rounds
        self.agreed     = False
        self.agreement  = None

    def is_acceptable(self, buyer_offer, seller_offer, tolerance=200):
        return abs(buyer_offer - seller_offer) <= tolerance

    def zopa(self):
        low  = max(self.buyer.offer, self.buyer.limit)
        high = min(self.seller.offer, self.seller.limit)
        if low <= high: return (low, high)
        return None

    def run(self):
        print(f"=== Salary Negotiation ===")
        print(f"  {self.buyer.name:>12}: initial=${self.buyer.offer:,.0f}  limit=${self.buyer.limit:,.0f}")
        print(f"  {self.seller.name:>12}: initial=${self.seller.offer:,.0f}  limit=${self.seller.limit:,.0f}")
        zopa = self.zopa()
        if zopa:
            print(f"  ZOPA: ${zopa[0]:,.0f}–${zopa[1]:,.0f}  (deal possible)\n")
        else:
            print("  No ZOPA — positions may not overlap\n")

        print(f"  {'Round':>6}  {'Candidate':>12}  {'Company':>10}  {'Gap':>10}  Status")
        print("  " + "─"*58)
        for rnd in range(1, self.max_rounds+1):
            candidate_offer = self.seller.counter_offer(self.buyer.offer, rnd, self.max_rounds)
            company_offer   = self.buyer.counter_offer(self.seller.offer, rnd, self.max_rounds)
            gap             = candidate_offer - company_offer
            status = "Agreed!" if self.is_acceptable(company_offer, candidate_offer) else "..."
            print(f"  {rnd:>6}  ${candidate_offer:>10,.0f}  ${company_offer:>8,.0f}  ${gap:>8,.0f}  {status}")
            if self.is_acceptable(company_offer, candidate_offer):
                self.agreed   = True
                self.agreement= (candidate_offer + company_offer) / 2
                break

        print()
        if self.agreed:
            print(f"  ✓ DEAL REACHED: ${self.agreement:,.0f}/year")
        else:
            final_gap = self.seller.offer - self.buyer.offer
            print(f"  ✗ No deal. Final gap: ${final_gap:,.0f}")
        return self.agreed, self.agreement

    def visualize(self, filename="negotiation.png"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        r = range(len(self.buyer.history))
        axes[0].plot(list(r), self.buyer.history,   "o-", color="steelblue", label=self.buyer.name)
        axes[0].plot(list(range(len(self.seller.history))), self.seller.history, "s-", color="tomato", label=self.seller.name)
        if self.agreed:
            axes[0].axhline(self.agreement, color="green", linestyle="--", linewidth=2, label=f"Deal=${self.agreement:,.0f}")
        axes[0].set_xlabel("Round"); axes[0].set_ylabel("Salary ($)"); axes[0].legend()
        axes[0].set_title("Negotiation Convergence"); axes[0].grid(alpha=0.3)
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}"))

        # Strategy comparison
        strategies = ["linear","aggressive","cooperative"]
        results = []
        for s in strategies:
            b=NegotiatorAgent("Company",  85000, 95000, 100000, s)
            sl=NegotiatorAgent("Candidate",120000,108000,105000, s)
            orch=NegotiationOrchestrator(b,sl,max_rounds=12)
            agreed,agr=orch.run()
            results.append((s, agr if agreed else None, b.history, sl.history))

        for i,(s,agr,bh,sh) in enumerate(results):
            axes[1].plot(range(len(bh)),bh,f"C{i}-",alpha=0.6)
            axes[1].plot(range(len(sh)),sh,f"C{i}--",alpha=0.6,label=f"{s} {'✓' if agr else '✗'}")
        axes[1].set_title("Strategy Comparison"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}"))

        plt.suptitle("Salary Negotiation Dynamics", fontsize=11)
        plt.tight_layout(); plt.savefig(filename, dpi=85); plt.close(); print(f"Saved {filename}")

print("=== Negotiation System ===\n")
candidate = NegotiatorAgent("Candidate", 120000, 108000, 105000, "cooperative")
company   = NegotiatorAgent("Company",    85000,  95000, 100000, "linear")
orch = NegotiationOrchestrator(candidate, company, max_rounds=12)
agreed, deal = orch.run()
orch.visualize()
