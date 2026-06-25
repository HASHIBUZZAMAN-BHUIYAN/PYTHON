# Day 03 — Solutions

# Exercise 1
age = int(input("Your age: "))
if age < 12:    price = 5
elif age < 18:  price = 8
elif age < 65:  price = 12
else:           price = 7
print(f"Ticket price: ${price}")

# Exercise 2
year = int(input("Year: "))
if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
    print("Leap year")
else:
    print("Not a leap year")

# Exercise 3
a = float(input("Side a: "))
b = float(input("Side b: "))
c = float(input("Side c: "))
if a + b > c and a + c > b and b + c > a:
    if a == b == c:     print("Equilateral")
    elif a == b or b == c or a == c: print("Isosceles")
    else:               print("Scalene")
else:
    print("Not a valid triangle")

# Exercise 4
USERNAME = "admin"
PASSWORD = "python123"
u = input("Username: ")
p = input("Password: ")
if u == USERNAME and p == PASSWORD: print("Login successful")
elif u != USERNAME and p != PASSWORD: print("Wrong username and password")
elif u != USERNAME: print("Wrong username")
else:               print("Wrong password")

# Exercise 5
n = float(input("Enter a number: "))
print("positive" if n > 0 else ("negative" if n < 0 else "zero"))
