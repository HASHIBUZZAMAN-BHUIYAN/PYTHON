# Day 04 — Solutions
import math

# Exercise 1
for i in range(1, 51):
    if i % 15 == 0:    print("FizzBuzz")
    elif i % 3 == 0:   print("Fizz")
    elif i % 5 == 0:   print("Buzz")
    else:              print(i)

# Exercise 2
values, total = [], 0
while True:
    raw = input("Enter a number (blank to stop): ")
    if raw == "": break
    values.append(float(raw))
    total += float(raw)
if values:
    print(f"Count: {len(values)}, Sum: {total}, Average: {total/len(values):.2f}")

# Exercise 3
n = int(input("Enter a number: "))
for d in range(2, math.isqrt(n) + 1):
    if n % d == 0:
        print(f"{n} is not prime (divisible by {d})")
        break
else:
    print(f"{n} is prime" if n > 1 else f"{n} is not prime")

# Exercise 4
n = int(input("Enter n: "))
for i in range(1, n + 1):
    print("*" * i)

# Exercise 5
SECRET = 42
for attempt in range(1, 6):
    guess = int(input(f"Attempt {attempt}/5 — guess: "))
    if guess == SECRET:
        print("Correct!")
        break
    elif guess < SECRET: print("Too low")
    else:                print("Too high")
else:
    print(f"Out of attempts. The answer was {SECRET}.")
