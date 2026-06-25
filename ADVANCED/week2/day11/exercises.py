# Advanced Day 11 — Exercises
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
np.random.seed(42); torch.manual_seed(42)

# Exercise 1 — Manual convolutions
# Apply the following kernels to a 10x10 image and visualize all results:
# a) Identity kernel (3x3 with 1 in center)
# b) Sharpen kernel [[0,-1,0],[-1,5,-1],[0,-1,0]]
# c) Gaussian blur (3x3 1/16 * [[1,2,1],[2,4,2],[1,2,1]])
# d) Sobel Y (edge detection in Y direction)
# Plot original + 4 filtered images side by side.
# TODO

# Exercise 2 — Receptive field
# After 2 conv layers (both 3×3, stride 1, no padding):
# What is the receptive field size on the original input?
# After 3×3 conv + maxpool(2,2) + 3×3 conv + maxpool(2,2):
# Starting from a 32×32 input — what is the spatial size at each layer?
# Compute manually and then verify with a PyTorch forward pass.
# TODO

# Exercise 3 — Depth-wise separable convolution
# Standard conv: in_c × out_c × kH × kW parameters
# Depthwise separable: in_c × kH × kW (depthwise) + in_c × out_c (pointwise)
# For in_c=32, out_c=64, k=3: compute parameter savings.
# Implement DepthwiseSepConv in PyTorch:
#   1. nn.Conv2d(in_c, in_c, 3, groups=in_c)   # depthwise
#   2. nn.Conv2d(in_c, out_c, 1)               # pointwise
# TODO

# Exercise 4 — CNN architecture from scratch (CIFAR-like synthetic)
# Generate synthetic 32x32x3 color images (10 classes, 500 per class).
# Build a 3-conv-layer CNN. Train for 5 epochs. Print accuracy.
# (NOTE: heavier — close other apps; ~500 MB RAM, ~5-10 min)
# TODO

# Exercise 5 — Visualize learned filters
# After training the SmallCNN from lesson.py, extract conv1.weight
# (shape: out_channels × 1 × kH × kW).
# Plot each filter as a grayscale image.
# TODO
