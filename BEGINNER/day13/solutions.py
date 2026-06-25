# Day 13 — Solutions
from collections import Counter
from datetime import date
import random, pathlib, sys, os

# Exercise 1
text = "the quick brown fox jumps over the lazy dog the fox"
char_freq = Counter(c for c in text if c != " ")
print("Top 3 chars:", char_freq.most_common(3))
word_freq = Counter(text.split())
print("Top 3 words:", word_freq.most_common(3))

# Exercise 2
def days_until(year, month, day):
    return (date(year, month, day) - date.today()).days

print(days_until(2026, 12, 31))

# Exercise 3
def deal_hand(n):
    ranks  = [str(r) for r in range(2,11)] + list("JQKA")
    suits  = ["♠","♥","♦","♣"]
    deck   = [r+s for s in suits for r in ranks]
    return random.sample(deck, n)

print(deal_hand(5))

# Exercise 4
p = pathlib.Path(__file__).parent
print("Dir:", p)
print("py files:", list(p.glob("*.py")))
tmp = p / "day13_temp"
tmp.mkdir(exist_ok=True)
(tmp / "test.txt").write_text("hello")
(tmp / "test.txt").unlink()
tmp.rmdir()
print("Temp folder created and removed.")

# Exercise 5 — string_tools.py is created below the solution code
# (see string_tools.py in day13/)
sys.path.insert(0, str(p))
# If string_tools.py exists, import and test
if (p / "string_tools.py").exists():
    import string_tools
    print(string_tools.reverse_words("hello world"))
    print(string_tools.is_anagram("listen","silent"))
    print(string_tools.word_count("the quick brown fox"))
