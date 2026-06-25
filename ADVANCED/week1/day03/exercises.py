# Advanced Day 03 — Exercises
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# Exercise 1 — Multi-line time series
# Generate 3 random walks (cumulative sums) of length 200.
# Plot all three on the same axes with different colors/labels.
# Add a horizontal line at y=0. Add legend. Save as "ex1.png".
# TODO

# Exercise 2 — Grouped bar chart
# Given the data below, create a side-by-side bar chart
# with groups on the x-axis and two bars per group.
subjects  = ["Math","Science","English","History"]
class_a   = [78, 82, 70, 88]
class_b   = [85, 77, 92, 74]
# Hint: x = np.arange(len(subjects)); bar width 0.35
# TODO

# Exercise 3 — Scatter with color mapping
# Generate 200 random points (x, y ~ N(0,1)).
# Compute distance from origin: d = sqrt(x^2 + y^2)
# Color each point by its distance using a colormap (e.g. "plasma").
# Add a colorbar.
# TODO

# Exercise 4 — Annotating a plot
# Plot y = x^2 for x in [-3, 3].
# Annotate the minimum point (0, 0) with an arrow and text.
# Shade the region where x > 1 in light red.
# TODO

# Exercise 5 — Custom style
# Recreate exercise 1 using plt.style.use("seaborn-v0_8-darkgrid") (or "ggplot").
# Compare how the style changes the appearance.
# TODO
