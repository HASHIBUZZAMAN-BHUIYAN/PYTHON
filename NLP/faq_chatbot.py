"""
FAQ Chatbot (Embedding Similarity)
====================================
What it does:
  A retrieval-based FAQ chatbot for a fictional SaaS product ("NovaDrive").
  Built-in list of 10 FAQ Q&A pairs. When a user asks a question, it:
    1. Encodes the question and all FAQ questions using sentence embeddings
       (all-MiniLM-L6-v2, ~90MB, fast on CPU)
    2. Computes cosine similarity between the query and all FAQ questions
    3. Returns the FAQ answer with the highest similarity score
    4. Shows a confidence indicator so the user knows how certain the match is

  Demo runs 6 test questions automatically to show the bot in action.

What it teaches:
  - Retrieval-based chatbot architecture (vs generative)
  - Semantic similarity: why embedding-based matching beats exact keyword match
    (e.g. "how do I cancel?" matches "account cancellation" even without exact words)
  - Confidence thresholding: when to say "I don't know" vs give an answer
  - How sentence transformers produce fixed-length vectors for any text

How to run:
  python NLP\faq_chatbot.py    (from PYTHON\ folder)

Estimated RAM: ~300MB (model load) | Time: ~2s after model is cached
Model: all-MiniLM-L6-v2 from sentence-transformers (~90MB download)
  Fast, CPU-friendly, widely used for semantic similarity tasks.
No API key needed.
"""

import numpy as np


# ─── FAQ KNOWLEDGE BASE ───────────────────────────────────────────────────────
# Fictional SaaS product: "NovaDrive" — cloud file management platform

FAQ_PAIRS = [
    {
        "q": "How do I reset my password?",
        "a": "Go to the login page and click 'Forgot Password'. Enter your email "
             "and you will receive a reset link within 2 minutes.",
    },
    {
        "q": "What payment methods do you accept?",
        "a": "We accept Visa, Mastercard, American Express, and PayPal. "
             "All payments are processed securely via Stripe.",
    },
    {
        "q": "How do I cancel my subscription?",
        "a": "You can cancel anytime from Settings > Billing > Cancel Subscription. "
             "Your access continues until the end of the current billing period.",
    },
    {
        "q": "Is my data backed up automatically?",
        "a": "Yes. NovaDrive backs up your data every 6 hours to three geographically "
             "separate data centres. You can restore any backup from the last 30 days.",
    },
    {
        "q": "How many users can share one account?",
        "a": "The Starter plan supports 1 user. The Team plan supports up to 15 users. "
             "The Enterprise plan has unlimited seats — contact sales for pricing.",
    },
    {
        "q": "What happens if I exceed my storage limit?",
        "a": "You will receive an email warning at 90% usage. At 100%, uploads pause "
             "until you upgrade your plan or delete old files.",
    },
    {
        "q": "Do you offer a free trial?",
        "a": "Yes, all plans come with a 14-day free trial. No credit card required "
             "to start. You are automatically moved to the free tier after the trial.",
    },
    {
        "q": "How do I share files with someone outside my team?",
        "a": "Click 'Share' on any file and choose 'Anyone with the link'. "
             "You can set an expiry date and require a password for sensitive files.",
    },
    {
        "q": "Is NovaDrive GDPR compliant?",
        "a": "Yes. We are fully GDPR compliant. Data is stored in EU data centres "
             "by default, and you can request a full data export or deletion at any time.",
    },
    {
        "q": "How do I contact customer support?",
        "a": "You can reach us via live chat (bottom-right bubble), email at "
             "support@novadrive.io, or phone Mon-Fri 9am-6pm GMT.",
    },
]


# ─── EMBEDDING MODEL ──────────────────────────────────────────────────────────

def load_model():
    from sentence_transformers import SentenceTransformer
    # all-MiniLM-L6-v2: 90MB, fast on CPU, excellent at semantic similarity
    return SentenceTransformer("all-MiniLM-L6-v2")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of unit-normalised vectors = cosine similarity."""
    norm_a = a / (np.linalg.norm(a) + 1e-9)
    norm_b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(norm_a, norm_b))


# ─── CHATBOT CLASS ────────────────────────────────────────────────────────────

class FAQChatbot:
    """
    Perceive  : user question (string)
    Reason    : embed question, compare to pre-computed FAQ embeddings
    Act       : return best-matching answer + confidence
    """

    LOW_CONFIDENCE  = 0.35   # below this: "I'm not sure"
    MED_CONFIDENCE  = 0.55   # above this: show answer with note
    HIGH_CONFIDENCE = 0.70   # above this: confident answer

    def __init__(self):
        print("  Loading sentence embedding model (all-MiniLM-L6-v2)...")
        self.model        = load_model()
        self.faq_qs       = [p["q"] for p in FAQ_PAIRS]
        self.faq_as       = [p["a"] for p in FAQ_PAIRS]
        # Pre-compute FAQ question embeddings at startup (fast retrieval later)
        self.faq_vecs     = self.model.encode(self.faq_qs, show_progress_bar=False)
        print(f"  FAQ indexed: {len(FAQ_PAIRS)} entries, "
              f"embedding dim: {self.faq_vecs.shape[1]}\n")

    def answer(self, question: str) -> dict:
        """Find the closest FAQ entry and return a structured result."""
        q_vec  = self.model.encode([question], show_progress_bar=False)[0]
        scores = [cosine_similarity(q_vec, fv) for fv in self.faq_vecs]
        best_i = int(np.argmax(scores))
        score  = scores[best_i]

        if score < self.LOW_CONFIDENCE:
            confidence_label = "LOW"
            reply = ("I'm not sure I have an answer for that. "
                     "Please contact support@novadrive.io for help.")
        elif score < self.MED_CONFIDENCE:
            confidence_label = "MEDIUM"
            reply = self.faq_as[best_i]
        else:
            confidence_label = "HIGH"
            reply = self.faq_as[best_i]

        return {
            "question":          question,
            "matched_faq":       self.faq_qs[best_i],
            "answer":            reply,
            "score":             score,
            "confidence":        confidence_label,
            "all_scores":        scores,
        }


# ─── DEMO ─────────────────────────────────────────────────────────────────────

TEST_QUESTIONS = [
    "I forgot my login credentials, how do I get back in?",    # -> password reset
    "Can I pay with PayPal?",                                  # -> payment methods
    "I want to stop my plan",                                  # -> cancel subscription
    "How safe is my information stored on your servers?",      # -> backups / GDPR
    "Is there a way to try before buying?",                    # -> free trial
    "What's your phone number?",                               # -> customer support
]

if __name__ == "__main__":
    print()
    print("=" * 68)
    print("  FAQ CHATBOT DEMO  (NovaDrive Support)")
    print("=" * 68)

    bot = FAQChatbot()

    for q in TEST_QUESTIONS:
        result = bot.answer(q)
        bar_len  = int(result["score"] * 20)
        bar      = "#" * bar_len + "." * (20 - bar_len)
        conf_tag = f"[{result['confidence']:<6}]"

        print(f"  Q: {q}")
        print(f"     Matched FAQ : {result['matched_faq']}")
        print(f"     Confidence  : {conf_tag} {result['score']:.3f}  [{bar}]")
        print(f"     Answer      : {result['answer'][:120]}")
        if len(result["answer"]) > 120:
            print(f"                   {result['answer'][120:200]}")
        print()

    print("=" * 68)
    print("  HOW IT WORKS:")
    print("  1. At startup: encode all FAQ questions into fixed-length vectors")
    print("  2. At query time: encode user question, compute cosine similarity")
    print("     with each FAQ vector, return the highest-scoring match")
    print("  3. Score > 0.70 = HIGH confidence | 0.55-0.70 = MEDIUM | <0.35 = LOW")
    print("  KEY INSIGHT: 'I forgot my login' matches 'How do I reset my password?'")
    print("  because embeddings capture MEANING, not just exact keywords.")
    print()
    print("[DONE] faq_chatbot.py complete")
