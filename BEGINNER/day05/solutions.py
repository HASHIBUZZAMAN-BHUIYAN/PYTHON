# Day 05 — Solutions

# Exercise 1
word = input("Enter a word: ").replace(" ", "").lower()
print("Palindrome" if word == word[::-1] else "Not a palindrome")

# Exercise 2
sentence = input("Enter a sentence: ")
words = sentence.lower().split()
freq = {}
for w in words:
    freq[w] = freq.get(w, 0) + 1
most_common = max(freq, key=freq.get)
print(f"Words: {len(words)}, Unique: {len(freq)}, Most common: '{most_common}' ({freq[most_common]}x)")

# Exercise 3
text = input("Enter text to encode: ")
result = []
for ch in text:
    if ch.isalpha():
        base = ord('A') if ch.isupper() else ord('a')
        result.append(chr((ord(ch) - base + 3) % 26 + base))
    else:
        result.append(ch)
print("".join(result))

# Exercise 4
raw = "  the quick BROWN fox  "
result = raw.strip().title()
print(result)

# Exercise 5
email = input("Enter email: ")
if email.count("@") != 1:
    print("Error: must contain exactly one @")
elif email.startswith("@") or email.endswith("@"):
    print("Error: @ cannot be first or last character")
elif "." not in email.split("@")[1]:
    print("Error: domain must contain a dot")
else:
    print("Valid")
