# Day 08 — Solutions
import math, time

# Exercise 1
def is_prime(n):
    if n < 2: return False
    for d in range(2, math.isqrt(n) + 1):
        if n % d == 0: return False
    return True

for n in [2, 3, 4, 17, 25, 97]:
    print(f"{n}: {is_prime(n)}")

# Exercise 2
def calc(*numbers, op="sum"):
    if op == "sum":     return sum(numbers)
    if op == "product":
        r = 1
        for n in numbers: r *= n
        return r
    if op == "mean":    return sum(numbers) / len(numbers)
    if op == "max":     return max(numbers)
    if op == "min":     return min(numbers)
    raise ValueError(f"Unknown op: {op}")

print(calc(1,2,3,4))
print(calc(1,2,3,4, op="product"))
print(calc(10,20,30, op="mean"))

# Exercise 3
def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

print(flatten([1, [2, [3, 4], 5], 6]))

# Exercise 4
def fib(n, memo={}):
    if n in memo: return memo[n]
    if n <= 1: return n
    memo[n] = fib(n-1, memo) + fib(n-2, memo)
    return memo[n]

print(fib(40))

# Exercise 5
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time()-start:.4f}s")
        return result
    return wrapper

@timer
def slow():
    time.sleep(0.1)
    return "done"

print(slow())
