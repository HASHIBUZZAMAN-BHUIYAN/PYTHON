# Day 08 Mini-Project — Math Toolkit
# A collection of useful math functions demonstrated interactively.

import math

def is_prime(n):
    if n < 2: return False
    for d in range(2, math.isqrt(n) + 1):
        if n % d == 0: return False
    return True

def primes_up_to(n):
    return [x for x in range(2, n+1) if is_prime(x)]

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

def factorial(n):
    return 1 if n <= 1 else n * factorial(n-1)

def fibonacci_seq(n):
    a, b, result = 0, 1, []
    for _ in range(n):
        result.append(a)
        a, b = b, a + b
    return result

menu = {
    "1": ("Primes up to N",           lambda: print(primes_up_to(int(input("N: "))))),
    "2": ("Check if number is prime", lambda: print(is_prime(int(input("N: "))))),
    "3": ("GCD of two numbers",       lambda: print(gcd(*[int(input(f"N{i+1}: ")) for i in range(2)]))),
    "4": ("LCM of two numbers",       lambda: print(lcm(*[int(input(f"N{i+1}: ")) for i in range(2)]))),
    "5": ("Factorial(n)",             lambda: print(factorial(int(input("N: "))))),
    "6": ("First N Fibonacci numbers",lambda: print(fibonacci_seq(int(input("N: "))))),
    "7": ("Quit",                     None),
}

while True:
    print("\n=== Math Toolkit ===")
    for k, (label, _) in menu.items():
        print(f"  {k}. {label}")
    ch = input("Choice: ").strip()
    if ch == "7":
        break
    if ch in menu:
        menu[ch][1]()
    else:
        print("Invalid choice.")
