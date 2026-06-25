# Day 04 — Loops

# ─── 1. for LOOP WITH range() ────────────────────────────────────────────────
for i in range(5):          # 0, 1, 2, 3, 4
    print(i, end=" ")
print()                     # newline

# range(start, stop, step)
for i in range(1, 11, 2):   # 1 3 5 7 9
    print(i, end=" ")
print()

# Count down
for i in range(10, 0, -1):
    print(i, end=" ")
print("Go!")

# ─── 2. for LOOP OVER SEQUENCES ──────────────────────────────────────────────
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit.upper())

# enumerate() gives (index, value)
for idx, fruit in enumerate(fruits, start=1):
    print(f"{idx}. {fruit}")

# zip() pairs two sequences
names = ["Alice", "Bob"]
ages  = [25, 30]
for name, age in zip(names, ages):
    print(f"{name} is {age}")

# ─── 3. while LOOP ───────────────────────────────────────────────────────────
count = 0
while count < 5:
    print(count, end=" ")
    count += 1
print()

# while with user input
while True:
    answer = input("Type 'quit' to exit: ")
    if answer.lower() == "quit":
        break
    print(f"You typed: {answer}")

# ─── 4. break AND continue ───────────────────────────────────────────────────
# break  – exit the loop immediately
for n in range(10):
    if n == 5:
        break           # stop at 5
    print(n, end=" ")
print()                 # 0 1 2 3 4

# continue – skip the rest of this iteration
for n in range(10):
    if n % 2 == 0:
        continue        # skip even numbers
    print(n, end=" ")
print()                 # 1 3 5 7 9

# ─── 5. for/while ... else ───────────────────────────────────────────────────
# The else block runs ONLY if the loop wasn't terminated by break.
for n in range(2, 10):
    if 9 % n == 0 and n != 9:
        print(f"9 is divisible by {n}")
        break
else:
    print("9 is prime")  # won't print (9 = 3*3)

# ─── 6. NESTED LOOPS ─────────────────────────────────────────────────────────
# Multiplication table 3x3
for row in range(1, 4):
    for col in range(1, 4):
        print(f"{row*col:3}", end="")
    print()

# ─── 7. LOOP PATTERNS ────────────────────────────────────────────────────────
# Accumulator
total = 0
for i in range(1, 101):
    total += i
print(f"Sum 1..100 = {total}")    # 5050

# Find first match
numbers = [4, 7, 2, 9, 1]
for n in numbers:
    if n > 6:
        print(f"First number > 6: {n}")
        break

# Collect into a list
squares = []
for i in range(1, 6):
    squares.append(i ** 2)
print(squares)   # [1, 4, 9, 16, 25]
