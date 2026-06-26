# Day 12 — OOP II

# ─── 1. INHERITANCE ──────────────────────────────────────────────────────────
class Animal:
    def __init__(self, name, sound):
        self.name  = name
        self.sound = sound

    def speak(self):
        return f"{self.name} says {self.sound}"

    def __str__(self):
        return f"Animal({self.name})"

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Woof")
        self.breed = breed

    def fetch(self):
        return f"{self.name} fetches the ball!"

    def __str__(self):
        return f"Dog({self.name}, {self.breed})"

class Cat(Animal):
    def __init__(self, name):
        super().__init__(name, "Meow")
        self.indoor = True

    def purr(self):
        return f"{self.name} purrs..."

fido  = Dog("Fido", "Labrador")
kitty = Cat("Kitty")

print(fido.speak())
print(fido.fetch())
print(kitty.speak())
print(kitty.purr())
print(fido)

# ─── 2. isinstance / issubclass ──────────────────────────────────────────────
print(isinstance(fido, Dog))
print(isinstance(fido, Animal))
print(issubclass(Dog, Animal))

# ─── 3. POLYMORPHISM ─────────────────────────────────────────────────────────
animals = [Dog("Rex","Poodle"), Cat("Whiskers"), Animal("Bird","Tweet")]
for a in animals:
    print(a.speak())

# ─── 4. ENCAPSULATION — name mangling ────────────────────────────────────────
class BankAccount:
    def __init__(self, owner, balance):
        self.owner   = owner
        self.__balance = balance    # name-mangled to _BankAccount__balance

    def deposit(self, amt):
        if amt > 0: self.__balance += amt

    @property
    def balance(self):          # read-only property
        return self.__balance

acct = BankAccount("Alice", 1000)
acct.deposit(500)
print(acct.balance)          # 1500 via property
# print(acct.__balance)      # AttributeError — name-mangled
print(acct._BankAccount__balance)  # works but "don't do this"

# ─── 5. DUNDER METHODS ───────────────────────────────────────────────────────
class Vector:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __add__(self, other):    return Vector(self.x+other.x, self.y+other.y)
    def __sub__(self, other):    return Vector(self.x-other.x, self.y-other.y)
    def __mul__(self, scalar):   return Vector(self.x*scalar,  self.y*scalar)
    def __rmul__(self, scalar):  return self.__mul__(scalar)   # 3*v works too
    def __eq__(self, other):     return self.x==other.x and self.y==other.y
    def __abs__(self):           return (self.x**2 + self.y**2) ** 0.5
    def __str__(self):           return f"Vector({self.x}, {self.y})"
    def __repr__(self):          return f"Vector({self.x!r}, {self.y!r})"

v1 = Vector(1, 2)
v2 = Vector(3, 4)
print(v1 + v2)   # Vector(4, 6)
print(v2 - v1)   # Vector(2, 2)
print(v1 * 3)    # Vector(3, 6)
print(3 * v1)    # Vector(3, 6)
print(abs(v2))   # 5.0
print(v1 == Vector(1, 2))  # True

# ─── 6. ABSTRACT BASE CLASSES (intro) ────────────────────────────────────────
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self): ...

    @abstractmethod
    def perimeter(self): ...

    def describe(self):
        print(f"{type(self).__name__}: area={self.area():.2f}, perimeter={self.perimeter():.2f}")

class Circle(Shape):
    def __init__(self, r): self.r = r
    def area(self):      return 3.14159 * self.r ** 2
    def perimeter(self): return 2 * 3.14159 * self.r

class Rect(Shape):
    def __init__(self, w, h): self.w, self.h = w, h
    def area(self):      return self.w * self.h
    def perimeter(self): return 2*(self.w+self.h)

for shape in [Circle(5), Rect(4, 6)]:
    shape.describe()
