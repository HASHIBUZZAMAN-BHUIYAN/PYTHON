# Day 01 — Solutions

# Exercise 1
my_name = "Hashib"
birth_year = 2000
my_height = 1.75
print(f"Name: {my_name}, Born: {birth_year}, Height: {my_height} m")

# Exercise 2
num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))
print(f"Sum:        {num1 + num2}")
print(f"Difference: {num1 - num2}")
print(f"Product:    {num1 * num2}")
print(f"Quotient:   {num1 / num2}")

# Exercise 3
p = 10
q = 20
p, q = q, p
print(f"After swap: p={p}, q={q}")

# Exercise 4
values = [42, 3.14, "hello", True, None]
for v in values:
    print(f"{v!r:10} -> {type(v)}")

# Exercise 5
celsius = float(input("Enter temperature in Celsius: "))
fahrenheit = celsius * 9 / 5 + 32
print(f"{celsius:.1f} °C = {fahrenheit:.1f} °F")
