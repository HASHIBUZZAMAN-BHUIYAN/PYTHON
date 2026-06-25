# Day 12 Mini-Project — RPG Character System (OOP II)
import math

class Character:
    def __init__(self, name, hp, attack, defense):
        self.name    = name
        self._hp     = hp
        self.max_hp  = hp
        self.attack  = attack
        self.defense = defense
        self.level   = 1
        self.xp      = 0

    @property
    def hp(self): return self._hp

    @property
    def alive(self): return self._hp > 0

    def take_damage(self, dmg):
        actual = max(0, dmg - self.defense)
        self._hp = max(0, self._hp - actual)
        return actual

    def heal(self, amount):
        self._hp = min(self.max_hp, self._hp + amount)

    def gain_xp(self, amount):
        self.xp += amount
        if self.xp >= self.level * 100:
            self.xp -= self.level * 100
            self.level += 1
            self.max_hp += 10; self._hp = self.max_hp
            self.attack  += 2; self.defense += 1
            print(f"  ** {self.name} leveled up to {self.level}! **")

    def attack_target(self, target):
        dmg = self.take_damage.__self__.attack if False else self.attack
        actual = target.take_damage(dmg)
        print(f"  {self.name} attacks {target.name} for {actual} dmg (HP: {target.hp}/{target.max_hp})")

    def __str__(self):
        bar_len = 20
        filled  = int(self._hp / self.max_hp * bar_len)
        bar     = "█" * filled + "░" * (bar_len - filled)
        return (f"{self.name} (Lv{self.level}) [{bar}] HP:{self._hp}/{self.max_hp} "
                f"ATK:{self.attack} DEF:{self.defense}")


class Warrior(Character):
    def __init__(self, name):
        super().__init__(name, hp=120, attack=15, defense=8)
    def shield_bash(self, target):
        dmg = self.attack + 5
        actual = target.take_damage(dmg)
        print(f"  {self.name} SHIELD BASHes {target.name} for {actual} dmg!")


class Mage(Character):
    def __init__(self, name):
        super().__init__(name, hp=80, attack=25, defense=3)
        self.mana = 100
    def fireball(self, target):
        if self.mana < 20:
            print(f"{self.name} is out of mana!")
            return
        self.mana -= 20
        dmg = self.attack * 2
        actual = target.take_damage(dmg)
        print(f"  {self.name} casts FIREBALL on {target.name} for {actual} dmg! (Mana:{self.mana})")


# --- Demo battle ---
hero    = Warrior("Arthur")
villain = Mage("Maleficent")

print("=== BATTLE START ===")
print(hero)
print(villain)
print()

for round_num in range(1, 6):
    print(f"--- Round {round_num} ---")
    hero.shield_bash(villain) if round_num % 2 == 0 else hero.attack_target(villain)
    if not villain.alive:
        print(f"{villain.name} is defeated!")
        hero.gain_xp(100)
        break
    villain.fireball(hero)
    if not hero.alive:
        print(f"{hero.name} is defeated!")
        break
    print(hero)
    print(villain)
    print()
