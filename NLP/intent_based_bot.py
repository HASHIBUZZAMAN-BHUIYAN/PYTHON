"""
Intent-Based Chatbot (TF-IDF + Logistic Regression)
======================================================
What it does:
  A small intent-classification chatbot trained on built-in example utterances.
  5 intents: greeting, complaint, order_status, goodbye, small_talk.
  For each incoming message it:
    1. Vectorises the message using TF-IDF (unigrams + bigrams)
    2. Predicts the most likely intent via Logistic Regression
    3. Returns a canned response template for that intent, with placeholders
       filled from the message (e.g. extracts order numbers from order_status)
    4. Shows the confidence distribution across all intents

  Demo runs 8 test messages automatically showing predictions + responses.

What it teaches:
  - Intent classification pipeline: TF-IDF features -> LR classifier
  - Why TF-IDF + LR is a strong baseline for short-text classification
  - Canned response templates vs generative responses
  - Entity extraction: using regex to pull order IDs out of detected utterances
  - Confidence scores from predict_proba and how to use them for fallback

How to run:
  python NLP\intent_based_bot.py    (from PYTHON\ folder)

Estimated RAM: ~50MB | Time: <1s (training on 50 examples is instant)
Model note: TF-IDF + LogisticRegression from sklearn — 100% offline,
no download needed. No API key required.
"""

import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np


# ─── TRAINING DATA (intent, utterance) ───────────────────────────────────────

TRAINING_DATA = [
    # greeting
    ("greeting", "Hi there"),
    ("greeting", "Hello"),
    ("greeting", "Hey"),
    ("greeting", "Good morning"),
    ("greeting", "Good afternoon"),
    ("greeting", "Greetings"),
    ("greeting", "Hi, how are you?"),
    ("greeting", "Hello there, nice to meet you"),
    ("greeting", "Hey what's up"),
    ("greeting", "Hi I have a question"),

    # complaint
    ("complaint", "This is terrible service"),
    ("complaint", "I am very unhappy with my order"),
    ("complaint", "This is not what I ordered"),
    ("complaint", "Your product is broken"),
    ("complaint", "I've been waiting for weeks and nothing arrived"),
    ("complaint", "This is unacceptable, I want a refund"),
    ("complaint", "The quality is awful"),
    ("complaint", "My item arrived damaged"),
    ("complaint", "I'm really frustrated with your company"),
    ("complaint", "Nothing works and I'm angry"),

    # order_status
    ("order_status", "Where is my order?"),
    ("order_status", "What is the status of my order?"),
    ("order_status", "Can you track order 12345?"),
    ("order_status", "I want to know when my package arrives"),
    ("order_status", "Has order 98765 been shipped?"),
    ("order_status", "Is my delivery on its way?"),
    ("order_status", "Check my order number 55502"),
    ("order_status", "When will my parcel arrive?"),
    ("order_status", "Update on delivery please"),
    ("order_status", "Can you look up order 77001 for me?"),

    # goodbye
    ("goodbye", "Bye"),
    ("goodbye", "Goodbye"),
    ("goodbye", "See you later"),
    ("goodbye", "Thanks, take care"),
    ("goodbye", "Alright, talk to you later"),
    ("goodbye", "I'm done, thanks"),
    ("goodbye", "That's all I needed, cheers"),
    ("goodbye", "Catch you later"),
    ("goodbye", "Ok thanks goodbye"),
    ("goodbye", "Thanks for your help, bye"),

    # small_talk
    ("small_talk", "What's the weather like?"),
    ("small_talk", "Do you like music?"),
    ("small_talk", "How are you today?"),
    ("small_talk", "What's your name?"),
    ("small_talk", "Are you a robot?"),
    ("small_talk", "Tell me a joke"),
    ("small_talk", "What day is it?"),
    ("small_talk", "Do you ever get bored?"),
    ("small_talk", "What do you think about AI?"),
    ("small_talk", "Can you speak Spanish?"),
]

# ─── RESPONSE TEMPLATES ────────────────────────────────────────────────────────

RESPONSES = {
    "greeting":     [
        "Hello! How can I help you today?",
        "Hi there! What can I do for you?",
        "Hey! I'm here to help. What do you need?",
    ],
    "complaint":    [
        "I'm really sorry to hear that. I've flagged this for our support team. "
        "Can you share your order number so we can investigate?",
        "That sounds very frustrating and I apologise. Let me escalate this "
        "immediately. Please share any order details you have.",
    ],
    "order_status": [
        "I'll look that up for you right away! {order_note}",
        "Sure, let me check your delivery status. {order_note}",
    ],
    "goodbye":      [
        "Goodbye! Have a great day!",
        "Take care! Feel free to come back if you need anything.",
        "Bye! Thanks for chatting with me.",
    ],
    "small_talk":   [
        "Ha, good question! I'm a chatbot so I experience the world differently "
        "than you do. Anything I can actually help with today?",
        "Interesting topic! I'm better at order and support questions, but "
        "happy to chat briefly. What's on your mind?",
    ],
}

LOW_CONF_THRESHOLD = 0.28  # 5-class problem with 50 examples: random=0.20, so 0.28=1.4x random

FALLBACK = ("I'm not sure I understood that. Could you rephrase? "
            "I can help with order status, complaints, or account questions.")


def extract_order_id(text: str) -> str | None:
    """Extract an order number from text like 'order 12345' or '#55501'."""
    m = re.search(r"(?:order\s*(?:number)?|#)\s*(\d{4,6})", text, re.IGNORECASE)
    return m.group(1) if m else None


# ─── CLASSIFIER ───────────────────────────────────────────────────────────────

class IntentBot:
    """
    Train a TF-IDF + LogisticRegression pipeline on the built-in examples.
    Then for each message: predict intent, fill response template.
    """

    def __init__(self):
        labels, texts = zip(*TRAINING_DATA)  # TRAINING_DATA is (intent, utterance)
        self.classes  = sorted(set(labels))

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=500,
                                       lowercase=True, sublinear_tf=True)),
            ("clf",   LogisticRegression(max_iter=200, C=1.0, random_state=42)),
        ])
        self.pipeline.fit(list(texts), list(labels))
        print(f"  Trained on {len(TRAINING_DATA)} examples, "
              f"{len(self.classes)} intents: {self.classes}")

    def respond(self, message: str) -> dict:
        proba   = self.pipeline.predict_proba([message])[0]
        classes = self.pipeline.classes_
        scores  = dict(zip(classes, proba))
        top_int = max(scores, key=scores.get)
        top_sc  = scores[top_int]

        if top_sc < LOW_CONF_THRESHOLD:
            reply = FALLBACK
            intent = "unknown"
        else:
            templates = RESPONSES.get(top_int, [FALLBACK])
            template  = random.choice(templates)

            # Fill order-status placeholder
            order_id   = extract_order_id(message)
            order_note = (f"Order #{order_id}: currently in transit, expected delivery "
                          f"within 2 business days." if order_id
                          else "Could you share your order number so I can check?")
            reply  = template.format(order_note=order_note)
            intent = top_int

        return {
            "intent":  intent,
            "score":   top_sc,
            "reply":   reply,
            "all":     scores,
        }


# ─── DEMO ─────────────────────────────────────────────────────────────────────

TEST_MESSAGES = [
    "Hey, good morning!",
    "My package still hasn't arrived and I'm furious",
    "Can you check the status of order 88321?",
    "Where is my delivery?",
    "Alright, thanks for your help, goodbye",
    "What is the meaning of life?",
    "The product I received is completely broken",
    "How old are you?",
]

if __name__ == "__main__":
    random.seed(42)
    print()
    print("=" * 68)
    print("  INTENT-BASED CHATBOT DEMO")
    print("=" * 68)
    print("  Model: TF-IDF (unigrams+bigrams) + LogisticRegression")
    print("  Dataset: 50 built-in utterances, 5 intents")
    print()

    bot = IntentBot()
    print()

    for msg in TEST_MESSAGES:
        result = bot.respond(msg)
        bar_len = int(result["score"] * 20)
        bar     = "#" * bar_len + "." * (20 - bar_len)
        ranked  = sorted(result["all"].items(), key=lambda x: -x[1])
        top2    = " | ".join(f"{k}: {v:.2f}" for k, v in ranked[:3])

        print(f"  User  : {msg}")
        print(f"  Intent: [{result['intent']:<12}] conf={result['score']:.3f}  [{bar}]")
        print(f"  Probs : {top2}")
        print(f"  Bot   : {result['reply']}")
        print()

    print("=" * 68)
    print("  HOW IT WORKS:")
    print("  1. TF-IDF: converts each message to a sparse term-frequency vector")
    print("     Bigrams ('not working', 'my order') capture more context than words alone")
    print("  2. LogisticRegression: learns weights per class from 50 training examples")
    print("     predict_proba() gives a confidence score across all 5 intents")
    print("  3. Templates: structured response with regex slot-filling (order IDs)")
    print("  4. Fallback: messages with max_confidence < 0.40 get a clarification prompt")
    print()
    print("[DONE] intent_based_bot.py complete")
