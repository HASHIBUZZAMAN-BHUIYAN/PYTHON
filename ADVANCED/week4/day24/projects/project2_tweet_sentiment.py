"""
Project: Tweet Sentiment Tracker
Teaches: short-text classification challenges, multi-class (pos/neg/neutral)
         sentiment, pie chart visualization of distribution.
~20 MB RAM, ~1s on CPU
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

# ─── 30 synthetic tweets (10 each class) ─────────────────────────────────────
TWEETS = [
    # Positive (label=2)
    "Just had the best coffee ever! ☕ Absolutely love mornings like this!",
    "Got my promotion today! Hard work really does pay off 🎉",
    "Beautiful sunset tonight, feeling so grateful for this view 🌅",
    "My team won the championship! Best day ever!! 🏆",
    "New puppy arrived! He is the cutest thing in the world 🐶",
    "Finally finished my first marathon! Incredible feeling 🏃",
    "Best birthday ever! Friends and family made it so special 🎂",
    "Nailed my presentation at work today, super proud of myself!",
    "Amazing concert last night, the energy was unreal 🎵",
    "Just got accepted into my dream university! Dreams come true 🎓",
    # Negative (label=0)
    "Flight delayed AGAIN. This airline is absolutely terrible 😠",
    "Laptop crashed and lost all my work. I want to cry 😭",
    "Traffic is unbearable. I've been stuck for 2 hours 🚗",
    "Customer service was so rude to me today, never going back",
    "Worst restaurant experience ever. Cold food, bad service 😤",
    "Just found out my wallet was stolen. Having the worst day",
    "Power cut right before my meeting. Technology hates me 💔",
    "Failed my driving test again. So frustrated right now",
    "Weather is miserable and my umbrella just broke 🌧️",
    "Woke up with a terrible headache after a bad night's sleep",
    # Neutral (label=1)
    "Heading to the supermarket to pick up some groceries.",
    "The meeting starts at 3pm. Room 204 on the second floor.",
    "Ordered a new book online. Should arrive by Thursday.",
    "Reminder: gym class is at 7am tomorrow morning.",
    "Just finished reading chapter 12 of my textbook.",
    "The bus leaves at 8:15. Make sure to be on time.",
    "Picked up coffee and the newspaper on my way to work.",
    "Working from home today. Setting up my desk now.",
    "Got a haircut this afternoon. Feels quite short.",
    "Checked the weather forecast. Might rain tomorrow.",
]
LABELS = [2]*10 + [0]*10 + [1]*10
CLASS_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}
COLORS      = {0: "tomato",   1: "steelblue", 2: "limegreen"}

# ─── Classify ─────────────────────────────────────────────────────────────────
vec  = TfidfVectorizer(ngram_range=(1,2), max_features=500)
X    = vec.fit_transform(TWEETS)
y    = np.array(LABELS)
lr   = LogisticRegression(max_iter=1000, C=1.0)
preds= cross_val_predict(lr, X, y, cv=3)
lr.fit(X, y)

# ─── Results ──────────────────────────────────────────────────────────────────
print("=== Tweet Sentiment Tracker ===\n")
print(f"{'Tweet':<60} {'True':>8}  {'Pred':>8}")
print("-"*80)
for tweet, true, pred in zip(TWEETS, LABELS, preds):
    match = "✓" if true==pred else "✗"
    print(f"{match} {tweet[:55]:<55} {CLASS_NAMES[true]:>8} → {CLASS_NAMES[pred]:>8}")

acc = np.mean(preds == y)
print(f"\nOverall accuracy: {acc:.1%}")

# ─── Distribution ─────────────────────────────────────────────────────────────
from collections import Counter
dist = Counter(preds)
labels_plot = [CLASS_NAMES[k] for k in sorted(dist.keys())]
sizes        = [dist[k] for k in sorted(dist.keys())]
colors_plot  = [COLORS[k] for k in sorted(dist.keys())]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].pie(sizes, labels=labels_plot, colors=colors_plot, autopct="%1.0f%%",
            startangle=140, textprops={"fontsize":11})
axes[0].set_title("Predicted Sentiment Distribution")

# Per-tweet confidence bar
probs = lr.predict_proba(X)
max_conf = probs.max(axis=1)
t_short  = [t[:20]+"…" for t in TWEETS]
bar_cols  = [COLORS[p] for p in preds]
axes[1].barh(range(len(TWEETS)), max_conf, color=bar_cols, alpha=0.8)
axes[1].axvline(0.5, color="k", linestyle="--", linewidth=0.8)
axes[1].set_yticks(range(len(TWEETS))); axes[1].set_yticklabels(t_short, fontsize=6)
axes[1].set_xlabel("Confidence"); axes[1].set_title("Per-Tweet Confidence")

plt.suptitle("Tweet Sentiment Tracker", fontsize=11)
plt.tight_layout(); plt.savefig("tweet_sentiment.png", dpi=85); plt.close()
print("Saved tweet_sentiment.png")
