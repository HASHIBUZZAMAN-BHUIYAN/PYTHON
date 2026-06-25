# Day 07 — Solutions

# Exercise 1
text = "the cat sat on the mat the cat sat"
freq = {}
for w in text.split():
    freq[w] = freq.get(w, 0) + 1
for word, count in sorted(freq.items(), key=lambda x: x[1], reverse=True):
    print(f"{word}: {count}")

# Exercise 2
book = {}
while True:
    cmd = input("add/find/delete/quit: ").strip().lower()
    if cmd == "add":
        n, num = input("Name: "), input("Number: ")
        book[n] = num
    elif cmd == "find":
        n = input("Name: ")
        print(book.get(n, "Not found"))
    elif cmd == "delete":
        n = input("Name: ")
        print("Deleted" if book.pop(n, None) else "Not found")
    elif cmd == "quit":
        break

# Exercise 3
original = {"a": 1, "b": 2, "c": 3, "d": 4}
inverted = {v: k for k, v in original.items()}
print(inverted)

# Exercise 4
class_a = {"Alice","Bob","Charlie","Diana","Eve"}
class_b = {"Charlie","Diana","Frank","Grace"}
print("Both:       ", class_a & class_b)
print("A only:     ", class_a - class_b)
print("Exactly one:", class_a ^ class_b)

# Exercise 5
words = ["apple","banana","ant","bee","cherry","avocado","blueberry","carrot"]
groups = {}
for w in words:
    key = w[0].lower()
    groups.setdefault(key, []).append(w)
for letter in sorted(groups):
    print(f"{letter}: {groups[letter]}")
