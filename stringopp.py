# ─────────────────────────────────────────────
#  SECTION 4: String Operations
# ─────────────────────────────────────────────

print("\n=== Strings ===")

sentence = "  Artificial Intelligence is the Future  "

print(sentence.strip())                    # remove whitespace
print(sentence.strip().lower())            # lowercase
print(sentence.strip().upper())            # uppercase
print(sentence.strip().split())            # split into list of words
print(sentence.strip().replace("Future", "Present"))
print(f"Length: {len(sentence.strip())}")
print(f"Starts with 'Art': {sentence.strip().startswith('Art')}")
print(f"Word count: {len(sentence.strip().split())}")

# Slicing — like cutting a tensor in deep learning
s = "DeepLearning"
print(f"\nFull: {s}")
print(f"First 4: {s[:4]}")
print(f"Last 8: {s[4:]}")
print(f"Every 2nd letter: {s[::2]}")
print(f"Reversed: {s[::-1]}")

# Join — used heavily in NLP tokenization
tokens = ["I", "love", "Python", "for", "AI"]
sentence_rebuilt = " ".join(tokens)
print(f"\nJoined: {sentence_rebuilt}")
