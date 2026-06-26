# Advanced Day 03 Mini-Project — Sales Dashboard
# Creates a 4-panel dashboard from synthetic sales data.
# ~80 MB RAM, <5s on CPU

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(99)

# ─── Synthetic data ───────────────────────────────────────────────────────────
months    = pd.date_range("2023-01-01", periods=12, freq="ME")
products  = ["Widget","Gadget","Doohickey"]
data_rows = []
for m in months:
    for p in products:
        units = np.random.randint(30, 150)
        price = {"Widget":9.99,"Gadget":24.99,"Doohickey":4.99}[p]
        data_rows.append({"month": m, "product": p,
                          "units": units, "revenue": units*price})

df = pd.DataFrame(data_rows)
monthly = df.groupby("month")["revenue"].sum()
product_rev = df.groupby("product")["revenue"].sum()
pivot = df.pivot_table(index="month", columns="product", values="revenue", aggfunc="sum")

# ─── Dashboard ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
fig.suptitle("2023 Sales Dashboard", fontsize=16, fontweight="bold")

gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)
ax1, ax2 = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1])
ax3, ax4 = fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])

ax1.plot(monthly.index, monthly.values/1000, marker="o", color="steelblue", linewidth=2)
ax1.fill_between(monthly.index, monthly.values/1000, alpha=0.15, color="steelblue")
ax1.set_title("Monthly Revenue (k$)"); ax1.set_xlabel("Month"); ax1.grid(True, alpha=0.3)

ax2.pie(product_rev, labels=product_rev.index, autopct="%1.1f%%", startangle=140)
ax2.set_title("Revenue by Product")

bottom = np.zeros(12)
colors = ["steelblue","tomato","seagreen"]
for prod, color in zip(products, colors):
    vals = pivot[prod].values / 1000
    ax3.bar(range(12), vals, bottom=bottom, label=prod, color=color, alpha=0.8)
    bottom += vals
ax3.set_title("Monthly Revenue by Product (k$)")
ax3.set_xticks(range(12))
ax3.set_xticklabels([m.strftime("%b") for m in months], rotation=45, fontsize=8)
ax3.legend(fontsize=8)

box_data = [df[df["product"]==p]["units"].values for p in products]
bp = ax4.boxplot(box_data, labels=products, patch_artist=True)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax4.set_title("Units Sold Distribution")

plt.savefig("sales_dashboard.png", dpi=90, bbox_inches="tight")
print("Saved sales_dashboard.png")
plt.show()
