"""
Project: Customer Support Chatbot
Teaches: retrieval bot with NER-based personalization, follow-up detection,
         session tracking and escalation logic.
~30 MB RAM, ~1s on CPU
"""
import re, random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque

# ─── Support KB ───────────────────────────────────────────────────────────────
SUPPORT_KB = [
    ("how do I reset my password",
     "To reset your password: visit login page → click 'Forgot Password' → enter email → check inbox."),
    ("my order has not arrived",
     "I'm sorry to hear that. Please provide your order number and I'll check the status immediately."),
    ("I want to return a product",
     "Returns are accepted within 30 days. Go to 'My Orders' → select item → click 'Return'. Free pickup."),
    ("how do I cancel my subscription",
     "To cancel: Account Settings → Subscription → Cancel Plan. Your access continues until the billing date."),
    ("I was charged twice for my order",
     "I apologize for the inconvenience. Please share your order ID and I'll escalate this to billing right away."),
    ("the product I received is damaged",
     "I'm sorry! Please send a photo to support@company.com with your order number and we'll send a replacement."),
    ("how do I change my delivery address",
     "You can change the address before shipment: My Orders → Edit Delivery Details. After dispatch, call support."),
    ("what payment methods are accepted",
     "We accept Visa, Mastercard, PayPal, Apple Pay, and bank transfer. Crypto is not currently supported."),
    ("how long does delivery take",
     "Standard delivery: 3-5 business days. Express: 1-2 days. International: 7-14 days."),
    ("I forgot my username",
     "Your username is the email address used to register. Try signing in with your email address directly."),
]

def preprocess(t):
    return re.sub(r"[^\w\s]", "", t.lower()).strip()

qs  = [preprocess(q) for q,_ in SUPPORT_KB]
ans = [a for _,a in SUPPORT_KB]
vec = TfidfVectorizer(ngram_range=(1,2), max_features=500)
Qmat= vec.fit_transform(qs)

ESCALATION_TRIGGERS = ["angry","furious","unacceptable","legal action","refund now","fraud","scam"]

class SupportBot:
    def __init__(self):
        self.session  = deque(maxlen=5)
        self.turn     = 0
        self.username = None

    def extract_name(self, text):
        m = re.search(r"(?:I am|my name is|this is)\s+([A-Z][a-z]+)", text)
        return m.group(1) if m else None

    def detect_escalation(self, text):
        return any(kw in text.lower() for kw in ESCALATION_TRIGGERS)

    def respond(self, user_input, threshold=0.1):
        self.turn += 1
        name = self.extract_name(user_input)
        if name: self.username = name

        greeting = f"Hi {self.username}! " if self.username else ""

        if self.detect_escalation(user_input):
            return f"{greeting}I understand your frustration. Escalating to a human agent now. Please hold."

        q_vec = vec.transform([preprocess(user_input)])
        sims  = cosine_similarity(q_vec, Qmat).flatten()
        best  = int(np.argmax(sims))
        conf  = float(sims[best])

        if conf < threshold:
            response = (f"{greeting}I'm not sure I understand. Could you rephrase? "
                        f"Or type 'human' to reach a live agent.")
        else:
            response = f"{greeting}{ans[best]}"

        self.session.append((user_input, response, conf))
        return response

bot = SupportBot()

CONVERSATION = [
    "Hi, my name is Sarah.",
    "I placed an order last week and it still hasn't arrived.",
    "My order number is ORD-12345.",
    "I also want to return another item I bought.",
    "The refund process is absolutely unacceptable! I am furious.",
    "How do I reset my password?",
    "How long does express delivery usually take?",
]

print("=== Customer Support Chatbot ===\n")
for msg in CONVERSATION:
    reply = bot.respond(msg)
    print(f"User: {msg}")
    print(f"Bot:  {reply}")
    print()

print(f"\n  Session length: {len(bot.session)} turns")
print("  Confidence per turn:")
for msg, resp, conf in bot.session:
    print(f"    [{conf:.3f}] {msg[:40]}")
