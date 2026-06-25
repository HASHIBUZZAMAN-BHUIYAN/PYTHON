# Advanced Day 18 — Exercises

# Exercise 1 — Custom edge detector
# Implement a Prewitt operator manually using cv2.filter2D.
# Prewitt X kernel: [[-1,0,1],[-1,0,1],[-1,0,1]]
# Prewitt Y kernel: [[-1,-1,-1],[0,0,0],[1,1,1]]
# Compute magnitude and compare with Sobel and Canny visually.
# TODO

# Exercise 2 — Template matching
# Create a small template image (crop a 40×40 region from scene.png).
# Use cv2.matchTemplate with TM_CCOEFF_NORMED to locate it in the full scene.
# Draw a rectangle around the best match.
# TODO

# Exercise 3 — Perspective transform (homography)
# Define a "top view" quadrilateral in the scene (any 4 points forming a skewed rect).
# Use cv2.getPerspectiveTransform to warp it to a 200×200 "bird's eye" view.
# This is used in robotics to correct camera tilt.
# TODO

# Exercise 4 — Object distance estimation (monocular)
# Assume a known-width object (e.g., the orange rectangle, real width = 0.3 m).
# Focal length f_pixels = 600 (given).
# Formula: distance = (real_width * f_pixels) / pixel_width
# Detect the orange rectangle in scene.png, measure its pixel width,
# and estimate the distance.
# TODO

# Exercise 5 — Simple SLAM: occupancy grid update
# Simulate a robot moving through a 20×20 grid.
# At each step, "scan" ±2 cells (mark as free) and if a wall is detected,
# mark cell as occupied.
# Start with all cells unknown (-1). Update to 0 (free) or 1 (occupied).
# Visualize the discovered map vs the true FLOOR map from Day 17.
# TODO
