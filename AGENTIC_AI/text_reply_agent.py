"""
Text Reply Agent (Customer Support)
====================================
What it does:
  Reads incoming customer messages, classifies the intent of each one
  (complaint, question, refund request, compliment, spam/abuse), then
  drafts a contextually appropriate reply using a tone rule-set:
    - complaint    -> empathetic + solution offer
    - question     -> concise factual answer from a mini-KB
    - refund       -> policy-aware, step-by-step instructions
    - compliment   -> warm acknowledgement
    - spam/abuse   -> flagged, NOT replied to

  Guardrail: toxic/abusive messages are quarantined. The agent prints
  a flag instead of generating a reply.

What it teaches:
  - Intent classification with keyword scoring (no ML library needed)
  - Rule-based tone selection and response templating
  - Input guardrails (spam/abuse detection before any processing)
  - Perceive -> classify -> guard -> respond agent loop

How to run:
  python text_reply_agent.py

API key needed? NO -- fully offline. No API key required.
  (A real deployment could swap mock_knowledge_lookup() for an LLM call.)
"""

import re
from dataclasses import dataclass
from typing import Optional


# ─── INTENT TAXONOMY ──────────────────────────────────────────────────────────

INTENT_KEYWORDS = {
    "complaint": [
        "terrible", "awful", "broken", "doesn't work", "not working",
        "disappointed", "waste", "useless", "horrible", "frustrated",
        "still waiting", "never received", "poor quality", "damaged",
    ],
    "refund": [
        "refund", "money back", "return", "cancel", "cancellation",
        "charge", "overcharged", "billing error", "dispute",
    ],
    "question": [
        "how do i", "how can i", "where is", "what is", "when will",
        "when does", "can i", "is it possible", "do you have", "?",
    ],
    "compliment": [
        "love", "amazing", "great", "excellent", "fantastic", "helpful",
        "wonderful", "thank you", "thanks", "best", "impressed", "happy",
    ],
    "spam": [
        "buy now", "click here", "limited offer", "free money", "winner",
        "congratulations you have won", "nigerian", "bitcoin", "crypto scheme",
    ],
    "abuse": [
        "idiot", "stupid", "hate you", "kill", "moron", "scam", "scammers",
        "shut down", "lawsuit", "sue you",
    ],
}

# Simple mini knowledge-base for the "question" intent
KNOWLEDGE_BASE = {
    "password":   "To reset your password, visit Settings > Account > Reset Password.",
    "shipping":   "Standard shipping takes 3-5 business days. Express takes 1-2 days.",
    "hours":      "Our support team is available Mon-Fri, 9am-6pm EST.",
    "return":     "Items can be returned within 30 days of purchase in original condition.",
    "account":    "You can manage your account at account.example.com.",
    "contact":    "Reach us at support@example.com or call 1-800-EXAMPLE.",
    "discount":   "Sign up for our newsletter to receive 10% off your first order.",
    "track":      "Use your order number at track.example.com to follow your shipment.",
}

# Response templates keyed by intent
TEMPLATES = {
    "complaint": (
        "Hi {name},\n\n"
        "I'm really sorry to hear about your experience - that's not the standard "
        "we hold ourselves to, and I completely understand your frustration.\n\n"
        "Here's what I'd like to do to make this right: {action}\n\n"
        "Please reply with your order number and I'll prioritise this for you.\n\n"
        "Sincerely,\nSupport Team"
    ),
    "refund": (
        "Hi {name},\n\n"
        "No problem - I'd be happy to help you with a refund.\n\n"
        "Here are the steps:\n"
        "  1. Log in to your account at account.example.com\n"
        "  2. Go to Orders > Select your order > Request Refund\n"
        "  3. Refunds are processed within 5-7 business days\n\n"
        "If you run into any trouble, just reply here and I'll handle it directly.\n\n"
        "Best,\nSupport Team"
    ),
    "question": (
        "Hi {name},\n\n"
        "Great question! {answer}\n\n"
        "Let me know if there's anything else I can help with.\n\n"
        "Best,\nSupport Team"
    ),
    "compliment": (
        "Hi {name},\n\n"
        "That truly made our day -- thank you so much! "
        "Feedback like yours motivates the whole team.\n\n"
        "We look forward to continuing to serve you.\n\n"
        "Warmly,\nSupport Team"
    ),
}


# ─── AGENT DATA TYPES ─────────────────────────────────────────────────────────

@dataclass
class InboundMessage:
    msg_id:  str
    sender:  str
    subject: str
    body:    str


@dataclass
class ClassificationResult:
    intent:     str
    confidence: float   # 0.0-1.0 (keyword hit rate)
    flags:      list    # guardrail flags, e.g. ["spam", "abuse"]


@dataclass
class Reply:
    to:      str
    subject: str
    body:    str
    action:  str        # "send" | "quarantine" | "escalate"


# ─── TEXT REPLY AGENT ─────────────────────────────────────────────────────────

class TextReplyAgent:
    """
    Perceive  : receive an InboundMessage
    Classify  : score against intent keywords; detect guardrail flags
    Guard     : if spam/abuse -> quarantine, do not reply
    Respond   : pick template by intent, fill slots, emit Reply
    """

    def __init__(self, company_name: str = "Acme Corp"):
        self.company = company_name
        self.processed: list[dict] = []   # audit log

    # ── Perceive ──────────────────────────────────────────────────────────────
    def perceive(self, msg: InboundMessage) -> str:
        """Combine subject + body for analysis."""
        return (msg.subject + " " + msg.body).lower()

    # ── Classify ──────────────────────────────────────────────────────────────
    def classify(self, text: str) -> ClassificationResult:
        """Score each intent bucket. Return highest-scoring intent + guardrail flags."""
        scores = {}
        flags  = []

        for intent, keywords in INTENT_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in text)
            scores[intent] = hits
            if intent in ("spam", "abuse") and hits > 0:
                flags.append(intent)

        # Remove guardrail intents from main ranking
        for flag in ("spam", "abuse"):
            scores.pop(flag, None)

        best_intent = max(scores, key=scores.get)
        best_score  = scores[best_intent]
        total_kws   = sum(len(v) for k, v in INTENT_KEYWORDS.items()
                          if k not in ("spam","abuse"))
        confidence  = min(1.0, best_score / max(1, total_kws * 0.1))

        return ClassificationResult(best_intent, confidence, flags)

    # ── Guard ─────────────────────────────────────────────────────────────────
    def guard(self, result: ClassificationResult, msg: InboundMessage) -> Optional[Reply]:
        """Return a quarantine Reply if flags are present; None means proceed."""
        if result.flags:
            return Reply(
                to      = "moderation-queue@example.com",
                subject = f"[QUARANTINED] {msg.subject}",
                body    = (f"Message from {msg.sender} flagged as: {', '.join(result.flags)}\n"
                           f"Original body:\n{msg.body[:200]}"),
                action  = "quarantine",
            )
        return None

    # ── Knowledge lookup for question intent ──────────────────────────────────
    def lookup(self, text: str) -> str:
        """Find the best matching KB entry, or return a generic fallback."""
        text_lower = text.lower()
        for keyword, answer in KNOWLEDGE_BASE.items():
            if keyword in text_lower:
                return answer
        return ("I'd be happy to help! Could you give me a bit more detail "
                "so I can point you to the right resource?")

    # ── Respond ───────────────────────────────────────────────────────────────
    def respond(self, msg: InboundMessage, result: ClassificationResult) -> Reply:
        """Fill the right template based on classified intent."""
        intent   = result.intent
        template = TEMPLATES.get(intent, TEMPLATES["question"])

        # Extract a first name (first word of sender, best effort)
        first_name = msg.sender.split()[0].title()

        slots = {"name": first_name, "action": "", "answer": ""}
        if intent == "complaint":
            slots["action"] = ("I'll personally look into this and get back "
                               "to you within 24 hours with a resolution.")
        elif intent == "question":
            slots["answer"] = self.lookup(msg.body)

        body = template.format(**slots)
        return Reply(
            to      = msg.sender,
            subject = f"Re: {msg.subject}",
            body    = body,
            action  = "send",
        )

    # ── Main loop ──────────────────────────────────────────────────────────────
    def process(self, msg: InboundMessage) -> Reply:
        """Full perceive -> classify -> guard -> respond cycle."""
        text        = self.perceive(msg)
        result      = self.classify(text)
        quarantine  = self.guard(result, msg)
        if quarantine:
            reply = quarantine
        else:
            reply = self.respond(msg, result)

        self.processed.append({
            "id":         msg.msg_id,
            "from":       msg.sender,
            "intent":     result.intent,
            "confidence": result.confidence,
            "flags":      result.flags,
            "action":     reply.action,
        })
        return reply


# ─── DEMO ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    MESSAGES = [
        InboundMessage(
            msg_id  = "MSG-001",
            sender  = "Sarah Thompson",
            subject = "Package arrived broken",
            body    = ("I received my order yesterday and the product is completely "
                       "broken. This is terrible. I'm so disappointed - it was a gift "
                       "and the quality is awful. I've never had such a bad experience."),
        ),
        InboundMessage(
            msg_id  = "MSG-002",
            sender  = "james",
            subject = "How do I reset my password?",
            body    = "Hi, how do I reset my password? I can't log in to my account.",
        ),
        InboundMessage(
            msg_id  = "MSG-003",
            sender  = "Maria Chen",
            subject = "Refund request",
            body    = ("I was overcharged on my last order. I'd like a refund please. "
                       "The billing error was $24.99 extra. Please cancel and refund."),
        ),
        InboundMessage(
            msg_id  = "MSG-004",
            sender  = "Alex Rivera",
            subject = "Just wanted to say thank you!",
            body    = ("Your support team is amazing. I love how quickly you responded. "
                       "Excellent service - best customer support I've ever had. Thank you!"),
        ),
        InboundMessage(
            msg_id  = "MSG-005",
            sender  = "unknown@sketchy.xyz",
            subject = "CONGRATULATIONS YOU HAVE WON",
            body    = ("Click here to claim your prize. Limited offer - free money. "
                       "Buy now before it expires. Bitcoin crypto scheme winner!"),
        ),
        InboundMessage(
            msg_id  = "MSG-006",
            sender  = "frustrated_user",
            subject = "When does my order ship?",
            body    = "When will my order ship? What is the standard shipping time?",
        ),
    ]

    agent = TextReplyAgent(company_name="Acme Corp")

    for msg in MESSAGES:
        reply = agent.process(msg)
        log   = agent.processed[-1]

        print("=" * 65)
        print(f"  MSG {msg.msg_id}  |  From: {msg.sender}")
        print(f"  Subject: {msg.subject}")
        print(f"  Detected intent : {log['intent'].upper():<12}  "
              f"confidence={log['confidence']:.0%}"
              + (f"  FLAGS={log['flags']}" if log['flags'] else ""))
        print(f"  Action          : {log['action'].upper()}")
        print()
        if reply.action == "quarantine":
            print(f"  [QUARANTINED -> {reply.to}]")
            print(f"  {reply.body[:120]}")
        else:
            # Print first 3 lines of reply body for brevity
            preview_lines = reply.body.split("\n")[:6]
            for line in preview_lines:
                print(f"  {line}")
            if len(reply.body.split("\n")) > 6:
                print("  [... reply continues ...]")
        print()

    print("=" * 65)
    print("  AUDIT LOG SUMMARY")
    print("  " + "-"*61)
    print(f"  {'ID':<10}  {'From':<22}  {'Intent':<12}  {'Action'}")
    for entry in agent.processed:
        print(f"  {entry['id']:<10}  {entry['from'][:20]:<22}  "
              f"{entry['intent']:<12}  {entry['action']}")
