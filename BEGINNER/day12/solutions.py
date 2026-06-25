# Day 12 — Solutions
import math

# Exercise 1
class Employee:
    def __init__(self, name, salary): self.name, self.salary = name, salary
    def annual_salary(self): return self.salary

class Manager(Employee):
    def __init__(self, name, salary, department):
        super().__init__(name, salary)
        self.department = department
        self.reports = []
    def add_report(self, emp): self.reports.append(emp)
    def annual_salary(self): return self.salary * 1.10

class Intern(Employee):
    def __init__(self, name, hourly_rate, hours_per_week):
        super().__init__(name, 0)
        self.hourly_rate = hourly_rate; self.hours_per_week = hours_per_week
    def annual_salary(self): return self.hourly_rate * self.hours_per_week * 52

for e in [Employee("Alice",50000), Manager("Bob",80000,"Engineering"), Intern("Carol",15,20)]:
    print(f"{e.name}: ${e.annual_salary():,.0f}")

# Exercise 2
class Fraction:
    def __init__(self, n, d):
        if d == 0: raise ZeroDivisionError("Denominator cannot be zero")
        g = math.gcd(abs(n), abs(d))
        sign = -1 if (n<0)^(d<0) else 1
        self.n, self.d = sign*abs(n)//g, abs(d)//g
    def __add__(self, o): return Fraction(self.n*o.d + o.n*self.d, self.d*o.d)
    def __sub__(self, o): return Fraction(self.n*o.d - o.n*self.d, self.d*o.d)
    def __mul__(self, o): return Fraction(self.n*o.n, self.d*o.d)
    def __truediv__(self,o): return Fraction(self.n*o.d, self.d*o.n)
    def __eq__(self,o):  return self.n==o.n and self.d==o.d
    def __lt__(self,o):  return self.n*o.d < o.n*self.d
    def __str__(self):   return f"{self.n}/{self.d}"
    def __repr__(self):  return f"Fraction({self.n},{self.d})"

a, b = Fraction(1,2), Fraction(1,3)
print(a+b, a-b, a*b, a/b)

# Exercise 3
class Circle:
    def __init__(self, radius): self.radius = radius
    @property
    def radius(self): return self._radius
    @radius.setter
    def radius(self, v):
        if v < 0: raise ValueError("Radius cannot be negative")
        self._radius = v
    @property
    def area(self):         return math.pi * self._radius**2
    @property
    def circumference(self):return 2*math.pi*self._radius

c = Circle(5)
print(c.area, c.circumference)
try: Circle(-1)
except ValueError as e: print(e)

# Exercise 4
class Flyable:
    def fly(self): return f"{self.name} is flying"
class Swimmable:
    def swim(self): return f"{self.name} is swimming"
class Animal:
    def __init__(self, name): self.name = name
class Duck(Animal, Flyable, Swimmable):
    pass

d = Duck("Donald")
print(d.fly(), d.swim())

# Exercise 5
class Countdown:
    def __init__(self, start): self.current = start
    def __iter__(self): return self
    def __next__(self):
        if self.current < 0: raise StopIteration
        val = self.current; self.current -= 1; return val

print(list(Countdown(5)))
