# Day 11 — Solutions

# Exercise 1
class Rectangle:
    def __init__(self, width, height):
        self.width, self.height = width, height
    def area(self):      return self.width * self.height
    def perimeter(self): return 2 * (self.width + self.height)
    def is_square(self): return self.width == self.height
    def __str__(self):   return f"Rectangle({self.width}x{self.height})"

r = Rectangle(4, 6)
print(r, r.area(), r.perimeter(), r.is_square())
print(Rectangle(5,5).is_square())

# Exercise 2
class Student:
    def __init__(self, name):
        self.name   = name
        self.grades = []
    def add_grade(self, s): self.grades.append(s)
    def average(self):  return sum(self.grades)/len(self.grades) if self.grades else 0
    def highest(self):  return max(self.grades)
    def lowest(self):   return min(self.grades)
    def __str__(self):  return f"{self.name}: avg={self.average():.1f}"

s = Student("Alice")
for g in [85,92,78,95]: s.add_grade(g)
print(s, "high:", s.highest(), "low:", s.lowest())

# Exercise 3
class Counter:
    total_counters = 0
    def __init__(self, step=1):
        self.count = 0; self.step = step
        Counter.total_counters += 1
    def increment(self): self.count += self.step
    def decrement(self): self.count -= self.step
    def reset(self):     self.count = 0
    def __str__(self):   return f"Counter({self.count}, step={self.step})"

c1, c2 = Counter(), Counter(5)
c1.increment(); c1.increment(); c2.increment()
print(c1, c2, "total:", Counter.total_counters)

# Exercise 4
class Temperature:
    def __init__(self, celsius): self.celsius = celsius
    def to_fahrenheit(self): return self.celsius * 9/5 + 32
    def to_kelvin(self):     return self.celsius + 273.15
    @classmethod
    def from_fahrenheit(cls, f): return cls((f - 32) * 5/9)
    def __str__(self): return f"{self.celsius:.1f} °C"

t = Temperature(100)
print(t, t.to_fahrenheit(), t.to_kelvin())
print(Temperature.from_fahrenheit(212))

# Exercise 5
class Stack:
    def __init__(self): self._data = []
    def push(self, item): self._data.append(item)
    def pop(self):
        if self.is_empty(): raise IndexError("Stack is empty")
        return self._data.pop()
    def peek(self):
        if self.is_empty(): raise IndexError("Stack is empty")
        return self._data[-1]
    def is_empty(self): return len(self._data) == 0
    def __len__(self):  return len(self._data)
    def __str__(self):  return f"Stack({self._data})"

st = Stack()
st.push(1); st.push(2); st.push(3)
print(st, len(st), st.peek())
print(st.pop(), st)
