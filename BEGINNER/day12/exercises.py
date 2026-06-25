# Day 12 — Exercises

# Exercise 1 — Employee hierarchy
# Base class: Employee(name, salary) with method annual_salary()
# Subclass: Manager(name, salary, department, reports=[])
#   add_report(emp), annual_salary() adds 10% bonus
# Subclass: Intern(name, hourly_rate, hours_per_week)
#   annual_salary() = hourly_rate * hours_per_week * 52
# Create instances of each, print their annual salaries.
# TODO

# Exercise 2 — Fraction class with dunder methods
# Fraction(num, den) — store in lowest terms (use gcd)
# Implement: +, -, *, /, ==, <, >, __str__ ("3/4"), __repr__
# TODO

# Exercise 3 — Property decorators
# Create a Circle class where:
# - radius is stored privately
# - Setting radius < 0 raises ValueError
# - area and circumference are read-only properties
# TODO

# Exercise 4 — Multiple inheritance
# Create Flyable (method fly()) and Swimmable (method swim()).
# Create Duck that inherits from both Animal (from lesson) and Flyable and Swimmable.
# TODO

# Exercise 5 — __iter__ and __next__
# Create a Countdown(start) class that is iterable.
# for n in Countdown(5): print(n)  → prints 5 4 3 2 1 0
# Implement __iter__ (returns self) and __next__ (raises StopIteration at -1)
# TODO
