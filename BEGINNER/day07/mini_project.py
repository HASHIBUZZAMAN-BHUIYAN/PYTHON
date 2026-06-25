# Day 07 Mini-Project — Simple Inventory System
# Uses a dict {item_name: {"qty": int, "price": float}}

inventory = {}

def show_menu():
    print("\n1.Add  2.Restock  3.Sell  4.View  5.Search  6.Quit")

while True:
    show_menu()
    ch = input("Choice: ").strip()
    if ch == "1":
        item  = input("Item name: ").lower()
        qty   = int(input("Quantity: "))
        price = float(input("Price per unit: $"))
        inventory[item] = {"qty": qty, "price": price}
        print(f"Added '{item}'.")
    elif ch == "2":
        item = input("Item to restock: ").lower()
        if item in inventory:
            inventory[item]["qty"] += int(input("Add quantity: "))
        else:
            print("Item not found.")
    elif ch == "3":
        item = input("Item to sell: ").lower()
        if item in inventory:
            n = int(input("Quantity to sell: "))
            if n <= inventory[item]["qty"]:
                inventory[item]["qty"] -= n
                print(f"Sold {n} × {item} = ${n * inventory[item]['price']:.2f}")
            else:
                print("Insufficient stock.")
        else:
            print("Item not found.")
    elif ch == "4":
        if not inventory:
            print("Empty inventory.")
        else:
            print(f"\n{'Item':<15} {'Qty':>6} {'Price':>8} {'Value':>10}")
            print("-" * 42)
            total = 0
            for item, d in sorted(inventory.items()):
                val = d['qty'] * d['price']
                total += val
                print(f"{item:<15} {d['qty']:>6} {d['price']:>8.2f} {val:>10.2f}")
            print(f"{'Total value':<31} {total:>10.2f}")
    elif ch == "5":
        q = input("Search: ").lower()
        results = {k: v for k, v in inventory.items() if q in k}
        if results:
            for k, v in results.items():
                print(f"  {k}: qty={v['qty']} price=${v['price']:.2f}")
        else:
            print("No matches.")
    elif ch == "6":
        print("Goodbye!")
        break
    else:
        print("Invalid choice.")
