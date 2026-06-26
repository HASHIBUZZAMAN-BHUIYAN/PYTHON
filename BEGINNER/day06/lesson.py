# Day 06 — Lists and Tuples

# ─── 1. CREATING LISTS ───────────────────────────────────────────────────────
nums   = [1, 2, 3, 4, 5]
mixed  = [42, "hello", 3.14, True, None]
empty  = []
nested = [[1, 2], [3, 4], [5, 6]]

# ─── 2. INDEXING & SLICING ───────────────────────────────────────────────────
fruits = ["apple", "banana", "cherry", "date", "elderberry"]
print(fruits[0])      # apple
print(fruits[-1])     # elderberry
print(fruits[1:4])    # ['banana', 'cherry', 'date']
print(fruits[::-1])   # reversed copy

# ─── 3. MUTATING LISTS ───────────────────────────────────────────────────────
fruits[1] = "blueberry"
print(fruits)

# ─── 4. LIST METHODS ─────────────────────────────────────────────────────────
numbers = [3, 1, 4, 1, 5, 9, 2, 6]

numbers.append(7)
print(numbers)

numbers.insert(0, 0)
print(numbers)

numbers.remove(1)          # remove first occurrence of value 1
print(numbers)

popped = numbers.pop()     # remove & return last element
print(popped, numbers)

numbers.sort()
print(numbers)

numbers.sort(reverse=True)
print(numbers)

copy = numbers.copy()
copy.reverse()
print(copy, numbers)       # copy reversed, original unchanged

print(numbers.count(1))
print(numbers.index(9))

numbers.clear()
print(numbers)

# ─── 5. LIST FUNCTIONS ───────────────────────────────────────────────────────
vals = [3, 1, 4, 1, 5, 9]
print(len(vals))        # 6
print(sum(vals))        # 23
print(min(vals))        # 1
print(max(vals))        # 9
print(sorted(vals))     # new sorted list, original unchanged

# ─── 6. ITERATING ────────────────────────────────────────────────────────────
for v in vals:
    print(v, end=" ")
print()

for i, v in enumerate(vals):
    print(f"[{i}]={v}", end="  ")
print()

# ─── 7. LIST COMPREHENSIONS ──────────────────────────────────────────────────
# [expression for item in iterable if condition]
squares = [x**2 for x in range(1, 6)]          # [1, 4, 9, 16, 25]
evens   = [x for x in range(20) if x % 2 == 0]
upper   = [w.upper() for w in ["hi", "bye"]]
print(squares, evens, upper)

# nested comprehension — flatten a 2D list
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [val for row in matrix for val in row]
print(flat)

# ─── 8. TUPLES ───────────────────────────────────────────────────────────────
# Tuples are IMMUTABLE lists. Use them for data that shouldn't change.
point = (3, 7)
rgb   = (255, 128, 0)
single = (42,)         # trailing comma required for single-element tuple!

print(point[0])         # 3
x, y = point            # unpacking
print(f"x={x}, y={y}")

# t[0] = 99  # ← TypeError — tuples are immutable

# Tuples as dictionary keys (lists can't be keys)
grid = {(0, 0): "origin", (1, 0): "right", (0, 1): "up"}
print(grid[(1, 0)])

# ─── 9. MEMBERSHIP TEST ──────────────────────────────────────────────────────
print("banana" in fruits)     # True
print(99 not in nums)         # True
