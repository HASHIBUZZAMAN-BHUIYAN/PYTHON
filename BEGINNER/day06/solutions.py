# Day 06 — Solutions

# Exercise 1
data = [12, 7, 3, 14, 6, 11, 5, 8, 9, 2]
s = sorted(data)
n = len(s)
mean   = sum(s) / n
median = (s[n//2-1] + s[n//2]) / 2 if n % 2 == 0 else s[n//2]
print(f"Min={min(s)} Max={max(s)} Mean={mean:.2f} Median={median} Range={max(s)-min(s)}")

# Exercise 2
items = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
unique_items = []
for x in items:
    if x not in unique_items:
        unique_items.append(x)
print(unique_items)

# Exercise 3
a = [x**2 for x in range(1, 21) if x % 2 != 0]
print(a)
words = ["cat", "elephant", "dog", "butterfly", "ant", "giraffe"]
b = [w.upper() for w in words if len(w) > 4]
print(b)

# Exercise 4
matrix = [[1,2,3],[4,5,6],[7,8,9]]
transposed = [[matrix[row][col] for row in range(3)] for col in range(3)]
print(transposed)

# Exercise 5
players = [("Alice", 88), ("Bob", 75), ("Charlie", 95), ("Diana", 82)]
ranked = sorted(players, key=lambda p: p[1], reverse=True)
print("\nLeaderboard:")
for rank, (name, score) in enumerate(ranked, 1):
    print(f"  {rank}. {name:<10} {score}")
