# Day 11 — OOP I: Classes and Objects

# ─── 1. DEFINING A CLASS ─────────────────────────────────────────────────────
class Dog:
    species = "Canis lupus familiaris"  # class attribute — shared by all instances

    def __init__(self, name, breed, age):
        self.name  = name
        self.breed = breed
        self.age   = age

    def bark(self):
        return f"{self.name} says: Woof!"

    def description(self):
        return f"{self.name} is a {self.age}-year-old {self.breed}"

    def __str__(self):
        return f"Dog({self.name}, {self.breed})"

    def __repr__(self):
        return f"Dog(name={self.name!r}, breed={self.breed!r}, age={self.age})"

# ─── 2. CREATING OBJECTS ─────────────────────────────────────────────────────
dog1 = Dog("Buddy", "Labrador", 3)
dog2 = Dog("Max",   "Poodle",   5)

print(dog1.name)
print(dog1.bark())
print(dog2.description())
print(dog1)
print(repr(dog1))

# ─── 3. CLASS vs INSTANCE ATTRIBUTES ─────────────────────────────────────────
print(dog1.species)
print(Dog.species)
Dog.species = "Domestic dog"
print(dog2.species)        # "Domestic dog" — all instances updated

# Instance attribute shadows class attribute if set on the instance
dog1.species = "Wolf hybrid"
print(dog1.species)
print(dog2.species)

# ─── 4. MODIFYING ATTRIBUTES ─────────────────────────────────────────────────
dog1.age = 4     # direct
print(dog1.age)

# ─── 5. MORE COMPLETE EXAMPLE — BankAccount ──────────────────────────────────
class BankAccount:
    interest_rate = 0.05

    def __init__(self, owner, balance=0.0):
        self.owner    = owner
        self.balance  = balance
        self._transactions = []   # _ prefix = convention for "private"

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self.balance += amount
        self._transactions.append(("deposit", amount))

    def withdraw(self, amount):
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount
        self._transactions.append(("withdraw", amount))

    def apply_interest(self):
        interest = self.balance * BankAccount.interest_rate
        self.deposit(interest)

    def statement(self):
        print(f"\n--- Statement for {self.owner} ---")
        for t_type, amt in self._transactions:
            print(f"  {t_type:8s}  ${amt:.2f}")
        print(f"  Balance: ${self.balance:.2f}")

    def __str__(self):
        return f"BankAccount({self.owner}, ${self.balance:.2f})"

acct = BankAccount("Alice", 1000)
acct.deposit(500)
acct.withdraw(200)
acct.apply_interest()
acct.statement()
print(acct)
