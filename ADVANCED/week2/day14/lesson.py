# Advanced Day 14 — Lesson: Review of Week 2 Concepts
# This lesson consolidates everything from days 8-13.

print("=== Week 2 Deep Learning — Concept Map ===\n")

concepts = {
    "Day 08 — NN from Scratch":
        "Backprop, gradient descent, activation functions — all manual with NumPy",
    "Day 09 — PyTorch":
        "Tensors, autograd, nn.Module, training loop, DataLoader",
    "Day 10 — TensorFlow/Keras":
        "Sequential/Functional API, compile(), fit(), callbacks, save/load",
    "Day 11 — CNN Fundamentals":
        "Conv layers, pooling, feature maps, building CNN in PyTorch",
    "Day 12 — CNN Tuning":
        "Data augmentation, BatchNorm, Dropout, LR schedules, ablation",
    "Day 13 — Transfer Learning":
        "Pretrained backbone, frozen weights, feature extraction, fine-tuning",
    "Day 14 — Capstone":
        "End-to-end pipeline using all Week 2 concepts",
}
for day, desc in concepts.items():
    print(f"  {day}")
    print(f"    → {desc}\n")

print("""
Key DL Checklist:
  ✓ Normalize input data
  ✓ Use BatchNorm for faster convergence
  ✓ Add Dropout to reduce overfitting
  ✓ Monitor train vs val accuracy — diverging = overfitting
  ✓ Use learning rate scheduler (CosineAnnealing, StepLR, ReduceLROnPlateau)
  ✓ Start with a pretrained backbone when data is small
  ✓ Use AdaptiveAvgPool2d for variable input sizes
  ✓ Save the best model (ModelCheckpoint or torch.save with val_acc check)
""")
