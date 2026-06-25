# Day 01 Mini-Project — Personal Profile Card
# Collects info from the user and prints a formatted profile card.

print("=== Personal Profile Card Generator ===\n")

name        = input("Full name: ")
age         = int(input("Age: "))
city        = input("City: ")
occupation  = input("Occupation/major: ")
fun_fact    = input("One fun fact about you: ")

birth_year = 2026 - age   # approximate

print("\n" + "=" * 40)
print("         PROFILE CARD")
print("=" * 40)
print(f"  Name       : {name}")
print(f"  Age        : {age}  (born ~{birth_year})")
print(f"  City       : {city}")
print(f"  Occupation : {occupation}")
print(f"  Fun fact   : {fun_fact}")
print("=" * 40)
print(f"\nHello {name.split()[0]}! Welcome to Python. 🐍")
