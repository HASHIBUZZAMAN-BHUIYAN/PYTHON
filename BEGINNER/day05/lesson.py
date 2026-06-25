# Day 05 — Strings

# ─── 1. CREATING STRINGS ─────────────────────────────────────────────────────
s1 = "Hello"
s2 = 'World'
s3 = """Multi
line"""
s4 = "It's fine"          # single quote inside double
s5 = 'Say "hello"'        # double quote inside single
print(s3)

# ─── 2. INDEXING ─────────────────────────────────────────────────────────────
word = "Python"
#        P  y  t  h  o  n
# index  0  1  2  3  4  5
#       -6 -5 -4 -3 -2 -1
print(word[0])     # P
print(word[-1])    # n  (last character)
print(len(word))   # 6

# ─── 3. SLICING  [start:stop:step] ───────────────────────────────────────────
# stop is EXCLUSIVE
print(word[0:3])   # Pyt
print(word[2:])    # thon
print(word[:4])    # Pyth
print(word[::2])   # Pto  (every 2nd char)
print(word[::-1])  # nohtyP  (reverse)

# ─── 4. STRINGS ARE IMMUTABLE ────────────────────────────────────────────────
# word[0] = 'p'  # ← TypeError! Can't change a character.
# To "change" it, build a new string:
new_word = 'p' + word[1:]
print(new_word)    # python

# ─── 5. CONCATENATION & REPETITION ──────────────────────────────────────────
greeting = "Hello" + ", " + "World!"
print(greeting)
print("ha" * 3)   # hahaha

# ─── 6. COMMON STRING METHODS ────────────────────────────────────────────────
text = "  Hello, Python World!  "
print(text.strip())          # removes leading/trailing whitespace
print(text.lower())
print(text.upper())
print(text.title())
print(text.replace("Python", "Amazing"))
print(text.find("Python"))   # index of first occurrence, -1 if not found
print("Python" in text)      # True — membership test

# split / join
sentence = "one two three"
words = sentence.split()         # split on whitespace → list
print(words)                     # ['one', 'two', 'three']
print("-".join(words))           # "one-two-three"

csv_line = "a,b,c,d"
parts = csv_line.split(",")
print(parts)                     # ['a', 'b', 'c', 'd']

# startswith / endswith
filename = "report.pdf"
print(filename.endswith(".pdf"))  # True

# ─── 7. F-STRINGS (detailed) ─────────────────────────────────────────────────
name = "Alice"
score = 98.567
print(f"Name: {name}")
print(f"Score: {score:.2f}")       # 2 decimal places
print(f"Score: {score:8.2f}")      # width 8, 2 decimals
print(f"Name: {name:>15}")         # right-align in 15 chars
print(f"Name: {name:<15}|")        # left-align
print(f"Name: {name:^15}|")        # center

# ─── 8. format() ─────────────────────────────────────────────────────────────
template = "Hello, {}! You scored {}."
print(template.format("Bob", 95))

# ─── 9. ESCAPE SEQUENCES ─────────────────────────────────────────────────────
print("Line1\nLine2")       # newline
print("Tab\there")          # tab
print("Quote: \"hi\"")      # escaped quote
print(r"Raw: \n stays \n")  # raw string — backslash not special
