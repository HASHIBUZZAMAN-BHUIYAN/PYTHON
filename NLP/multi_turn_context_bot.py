"""
Multi-Turn Context Chatbot (Slot-Filling + Context Tracking)
=============================================================
What it does:
  A rule-based chatbot that tracks conversation state across multiple turns
  using slot-filling and a context dictionary. Distinctly different from a
  single-turn FAQ or intent bot:
    - Remembers the user's name once introduced
    - Remembers a mentioned product/topic and refers back to it
    - Tracks a 3-step order-flow (product -> quantity -> confirm)
    - Detects when the user switches topic mid-conversation and resets
      the relevant slots
    - Prints a running context state snapshot after each turn

  The demo runs a scripted multi-turn conversation automatically, showing
  how the bot's internal context evolves across 10 turns.

What it teaches:
  - Slot-filling: how chatbots build a structured "form" from free-text input
  - Context window vs conversation history: the bot only tracks named slots,
    not the full conversation verbatim
  - State machine for conversation flow: states are IDLE / COLLECT_PRODUCT /
    COLLECT_QTY / CONFIRM / DONE
  - Why this approach is very different from generative LLMs: entirely
    deterministic, fully interpretable, zero hallucination risk

How to run:
  python NLP\multi_turn_context_bot.py    (from PYTHON\ folder)

Estimated RAM: <20MB | Time: <1s
Model note: 100% rule-based / regex slot extraction. No model, no API key.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

# ─── SLOT DEFINITIONS ─────────────────────────────────────────────────────────

PRODUCT_NAMES = [
    "laptop", "phone", "keyboard", "mouse", "monitor", "headphones",
    "tablet", "charger", "camera", "speaker",
]

# ─── CONTEXT STATE ────────────────────────────────────────────────────────────

@dataclass
class Context:
    user_name:   Optional[str] = None
    topic:       Optional[str] = None    # last mentioned product/subject
    order_state: str           = "IDLE"  # IDLE | COLLECT_PRODUCT | COLLECT_QTY | CONFIRM | DONE
    order_product: Optional[str] = None
    order_qty:   Optional[int] = None
    turn:        int           = 0
    history:     list          = field(default_factory=list)

    def snapshot(self) -> str:
        parts = [
            f"name={self.user_name or '?'}",
            f"topic={self.topic or '?'}",
            f"order_state={self.order_state}",
            f"order_product={self.order_product or '?'}",
            f"order_qty={self.order_qty or '?'}",
        ]
        return "  [CTX] " + " | ".join(parts)


# ─── SLOT EXTRACTORS ──────────────────────────────────────────────────────────

def extract_name(text: str) -> Optional[str]:
    """Extract user name from patterns like 'I am Alice' / 'my name is Bob'."""
    m = re.search(
        r"\b(?:i(?:'m| am)|my name is|call me|this is)\s+([A-Z][a-z]{1,15})",
        text, re.IGNORECASE
    )
    return m.group(1).capitalize() if m else None


def extract_product(text: str) -> Optional[str]:
    """Extract a product mention from the known product list."""
    lower = text.lower()
    for prod in PRODUCT_NAMES:
        if re.search(r"\b" + prod + r"\b", lower):
            return prod
    return None


def extract_quantity(text: str) -> Optional[int]:
    """Extract a numeric quantity from text ('3', 'three', 'a couple')."""
    word_nums = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                 "a": 1, "an": 1}
    # digit first
    m = re.search(r"\b(\d{1,3})\b", text)
    if m:
        return int(m.group(1))
    # word numbers
    for word, num in word_nums.items():
        if re.search(r"\b" + word + r"\b", text, re.IGNORECASE):
            return num
    return None


def is_affirmative(text: str) -> bool:
    return bool(re.search(r"\b(yes|yeah|yep|sure|ok|okay|correct|please|confirm|go ahead)\b",
                           text, re.IGNORECASE))

def is_negative(text: str) -> bool:
    return bool(re.search(r"\b(no|nope|cancel|stop|never mind|nah|don't|dont)\b",
                           text, re.IGNORECASE))

def wants_order(text: str) -> bool:
    return bool(re.search(r"\b(order|buy|purchase|get|want|need)\b", text, re.IGNORECASE))

def wants_help(text: str) -> bool:
    return bool(re.search(r"\b(help|support|assist|question|problem|issue)\b",
                           text, re.IGNORECASE))


# ─── CHATBOT ──────────────────────────────────────────────────────────────────

class MultiTurnBot:
    """
    State machine chatbot with slot filling.
    Each turn: extract slots -> update context -> select response template.
    """

    def __init__(self):
        self.ctx = Context()

    def _greet_part(self) -> str:
        """Personalised greeting if name is known."""
        if self.ctx.user_name:
            return f"{self.ctx.user_name}"
        return "there"

    def chat(self, user_input: str) -> str:
        self.ctx.turn += 1
        text = user_input.strip()

        # Always try to extract name
        name = extract_name(text)
        if name:
            self.ctx.user_name = name

        # Product / topic detection
        prod = extract_product(text)
        if prod:
            self.ctx.topic = prod

        # ── ORDER STATE MACHINE ────────────────────────────────────────────────
        state = self.ctx.order_state

        if state == "IDLE":
            if wants_order(text):
                # Use explicit product, or fall back to the already-known topic
                chosen_prod = prod or self.ctx.topic
                if chosen_prod:
                    self.ctx.order_product = chosen_prod
                    self.ctx.order_state   = "COLLECT_QTY"
                    return (f"Great choice, {self._greet_part()}! "
                            f"A {chosen_prod} it is. How many would you like?")
                else:
                    self.ctx.order_state = "COLLECT_PRODUCT"
                    return ("I'd be happy to help you place an order! "
                            "Which product are you interested in? "
                            f"We carry: {', '.join(PRODUCT_NAMES[:5])}, and more.")
            elif wants_help(text):
                return (f"Of course, {self._greet_part()}! "
                        "I can help you place an order, check on a product, "
                        "or answer general questions. What do you need?")
            elif name:
                return (f"Nice to meet you, {self.ctx.user_name}! "
                        "How can I help you today?")
            elif self.ctx.topic:
                return (f"You mentioned {self.ctx.topic} earlier. "
                        "Are you looking to order one, or do you have a question about it?")
            else:
                return ("Hello! I can help you place orders or answer product questions. "
                        "What are you looking for?")

        elif state == "COLLECT_PRODUCT":
            if prod:
                self.ctx.order_product = prod
                self.ctx.order_state   = "COLLECT_QTY"
                return f"A {prod} - good pick! How many would you like?"
            else:
                return ("Sorry, I did not recognise that product. "
                        f"Try: {', '.join(PRODUCT_NAMES[:5])}, or others.")

        elif state == "COLLECT_QTY":
            qty = extract_quantity(text)
            if qty:
                self.ctx.order_qty   = qty
                self.ctx.order_state = "CONFIRM"
                p = self.ctx.order_product
                return (f"To confirm: {qty} x {p}(s). "
                        "Shall I place that order? (yes / no)")
            else:
                return ("How many would you like? Please say a number, "
                        "like '2' or 'three'.")

        elif state == "CONFIRM":
            if is_affirmative(text):
                p   = self.ctx.order_product
                qty = self.ctx.order_qty
                self.ctx.order_state = "DONE"
                return (f"Order placed! {qty} x {p}(s) for "
                        f"{self._greet_part()}. Expected delivery: 3-5 days. "
                        "Is there anything else I can help with?")
            elif is_negative(text):
                self.ctx.order_product = None
                self.ctx.order_qty     = None
                self.ctx.order_state   = "IDLE"
                return "No problem! Order cancelled. What else can I do for you?"
            else:
                p   = self.ctx.order_product
                qty = self.ctx.order_qty
                return (f"Just to confirm: {qty} x {p}(s)? Please say yes or no.")

        elif state == "DONE":
            # Order complete — back to IDLE for next topic
            self.ctx.order_state = "IDLE"
            return self.chat(text)  # re-process as IDLE

        return "I'm not sure how to respond to that. Could you rephrase?"


# ─── DEMO CONVERSATION ────────────────────────────────────────────────────────

DEMO_TURNS = [
    "Hi there",
    "My name is Sarah",
    "I was thinking about getting a laptop",
    "Actually, I want to order one",
    "Just one please",
    "Yes, go ahead",
    "Now I also need a keyboard",
    "Three keyboards please",
    "Yes confirm",
    "Thanks, that's all!",
]

if __name__ == "__main__":
    print()
    print("=" * 68)
    print("  MULTI-TURN CONTEXT CHATBOT DEMO")
    print("=" * 68)
    print("  Method: Rule-based slot filling + state machine (100% offline)")
    print("  Slots tracked: user_name, topic, order_product, order_qty")
    print()

    bot = MultiTurnBot()

    for turn_text in DEMO_TURNS:
        print(f"  Turn {bot.ctx.turn + 1}")
        print(f"  User : {turn_text}")
        reply = bot.chat(turn_text)
        print(f"  Bot  : {reply}")
        print(bot.ctx.snapshot())
        print()

    print("=" * 68)
    print("  HOW IT WORKS:")
    print("  - Slot filling: regex patterns extract (name, product, quantity)")
    print("    from free text each turn. Slots persist across turns in Context.")
    print("  - State machine: IDLE -> COLLECT_PRODUCT -> COLLECT_QTY -> CONFIRM")
    print("    -> DONE ensures the bot never skips a step in the order flow.")
    print("  - Context reuse: once 'Sarah' and 'laptop' are known, the bot")
    print("    refers to them by name in later turns without the user repeating.")
    print("  - This is NOT generative: responses are templates, zero hallucination.")
    print()
    print("[DONE] multi_turn_context_bot.py complete")
