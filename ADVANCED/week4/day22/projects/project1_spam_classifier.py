# ~200 MB RAM, ~8s on CPU
"""
Project 1: Spam Classifier
===========================
What it does: Classifies SMS/email messages as spam or ham (not spam)
              using TF-IDF features + Logistic Regression.

What it teaches:
  - Building a text classification pipeline end-to-end
  - TF-IDF vectorization with sklearn
  - Evaluating classifiers with accuracy + confusion matrix
  - Generating synthetic training data for NLP
"""

import re
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

random.seed(42)

# ─── Synthetic Dataset (200 messages) ───────────────────────────────────────

SPAM_TEMPLATES = [
    "WINNER! You have won a ${amt} prize. Click here to claim now!",
    "FREE offer! Get ${amt} cash reward. Call {phone} immediately!",
    "Congratulations! You've been selected for a ${amt} bonus. Reply YES",
    "URGENT: Your account will be closed. Verify now at {url}",
    "You have {num} unclaimed messages. Click {url} to view them now",
    "Win BIG! Enter now and get ${amt} instantly. Limited time offer!",
    "Your loan of ${amt} is approved! Call {phone} to collect today",
    "FREE ringtones! Text {code} to {phone}. Only ${amt}/week",
    "Alert: Suspicious login detected. Confirm identity at {url}",
    "Claim your free iPhone! You've been chosen. Reply NOW to win",
    "SALE 90% OFF today only! Order at {url} before midnight",
    "You owe ${amt} in taxes. Pay immediately or face legal action",
    "Hot singles in your area! Call {phone} for fun tonight",
    "Make ${amt} per week working from home! No experience needed",
    "Your parcel #${num} could not be delivered. Click {url} to reschedule",
    "Exclusive deal for you: ${amt} gift card. Claim in {num} hours",
    "Double your money fast! Investment opportunity at {url}",
    "You are the lucky winner! Send your details to {phone}",
    "FREE credit check! No obligation. Visit {url} now",
    "Earn ${amt} per survey. Sign up free at {url} today",
]

HAM_TEMPLATES = [
    "Hey, are you free for lunch tomorrow? Let me know!",
    "The meeting has been moved to 3pm. See you there.",
    "Can you pick up some milk on your way home?",
    "Happy birthday! Hope you have a wonderful day.",
    "I'll be 10 minutes late. Start without me.",
    "Did you see the game last night? Amazing ending!",
    "Thanks for your help yesterday. Really appreciate it.",
    "Reminder: doctor appointment on Thursday at 2pm.",
    "The project deadline has been extended to next Friday.",
    "Just finished the report. Sending it over now.",
    "Are you coming to the party on Saturday?",
    "I left my keys at your place. Can I come by to get them?",
    "The wifi password is {code}. Connect and join the call.",
    "Mom called. She wants you to call her back when you can.",
    "Flight departs at {num}am. Don't forget your passport.",
    "Your package has been delivered to the front door.",
    "Can we reschedule our call to {num}pm? Something came up.",
    "Great work on the presentation! Everyone was impressed.",
    "The restaurant is booked for {num} people at 7pm.",
    "I sent you the file. Let me know if you can open it.",
]

def fill_template(tmpl):
    return (tmpl
            .replace("{amt}",   str(random.randint(100, 9999)))
            .replace("{num}",   str(random.randint(1, 99)))
            .replace("{phone}", f"0{random.randint(7000000000, 7999999999)}")
            .replace("{url}",   f"http://bit.ly/{random.randint(10000,99999)}")
            .replace("{code}",  f"CODE{random.randint(100,999)}"))

messages = []
labels   = []

for _ in range(100):
    tmpl = random.choice(SPAM_TEMPLATES)
    messages.append(fill_template(tmpl))
    labels.append(1)

for _ in range(100):
    tmpl = random.choice(HAM_TEMPLATES)
    messages.append(fill_template(tmpl))
    labels.append(0)

combined = list(zip(messages, labels))
random.shuffle(combined)
messages, labels = zip(*combined)

print("=" * 60)
print("SPAM CLASSIFIER — TF-IDF + Logistic Regression")
print("=" * 60)
print(f"Dataset: {len(messages)} messages ({sum(labels)} spam, {len(labels)-sum(labels)} ham)")
print()

# ─── Sample Messages ────────────────────────────────────────────────────────
print("Sample messages:")
for i in range(3):
    print(f"  [{'SPAM' if labels[i]==1 else 'HAM ':4}] {messages[i][:70]}")
print()

# ─── Train/Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    messages, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"Train: {len(X_train)} messages | Test: {len(X_test)} messages")

# ─── TF-IDF Vectorization ───────────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),    # unigrams + bigrams
    max_features=3000,
    sublinear_tf=True,     # use log(1+tf) scaling
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)
print(f"Vocabulary size: {len(vectorizer.vocabulary_)} features")

# ─── Train Logistic Regression ──────────────────────────────────────────────
clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
clf.fit(X_train_vec, y_train)

# ─── Evaluate ───────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test_vec)
acc    = accuracy_score(y_test, y_pred)
cm     = confusion_matrix(y_test, y_pred)

print(f"\nAccuracy: {acc:.1%}")
print("\nConfusion Matrix:")
print(f"              Predicted HAM  Predicted SPAM")
print(f"  Actual HAM       {cm[0][0]:>4}            {cm[0][1]:>4}")
print(f"  Actual SPAM      {cm[1][0]:>4}            {cm[1][1]:>4}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# ─── Top Spam-Indicating Words ──────────────────────────────────────────────
feature_names = vectorizer.get_feature_names_out()
coef = clf.coef_[0]
top_spam_idx = np.argsort(coef)[-10:][::-1]
top_ham_idx  = np.argsort(coef)[:10]

print("Top 10 SPAM-indicating words:")
for idx in top_spam_idx:
    print(f"  {feature_names[idx]:<25} coef={coef[idx]:+.3f}")

print("\nTop 10 HAM-indicating words:")
for idx in top_ham_idx:
    print(f"  {feature_names[idx]:<25} coef={coef[idx]:+.3f}")

# ─── Live Prediction Demo ────────────────────────────────────────────────────
print("\nLive Predictions on new messages:")
new_messages = [
    "WINNER! You have been selected for a $5000 prize! Call now!",
    "Hey, can we meet at 3pm to discuss the project?",
    "FREE iPhone! Click here to claim your exclusive offer today",
    "Reminder: your dentist appointment is tomorrow at 10am",
    "URGENT: Verify your account or it will be suspended immediately",
]

for msg in new_messages:
    vec  = vectorizer.transform([msg])
    pred = clf.predict(vec)[0]
    prob = clf.predict_proba(vec)[0]
    label = "SPAM" if pred == 1 else "HAM "
    conf  = prob[pred]
    print(f"  [{label}] conf={conf:.1%} | {msg[:60]}")

print("\n--- Project 1 Complete ---")
