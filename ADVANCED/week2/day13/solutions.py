# Advanced Day 13 — Solutions (Conceptual / Outline — full code would be very long)
# These sketches show the approach; the full runnable versions are in exercises.py

import torch, torch.nn as nn

print("=== Exercise 1 — Fine-tuning outline ===")
print("""
# After head training:
for name, param in backbone.named_parameters():
    if "features.17" in name or "features.18" in name:  # last 2 feature blocks
        param.requires_grad = True

optimizer = torch.optim.Adam([
    {"params": backbone.classifier.parameters(), "lr": 1e-3},
    {"params": [p for p in backbone.features.parameters() if p.requires_grad], "lr": 1e-4}
])
# Then continue training loop for more epochs.
""")

print("=== Exercise 2 — Feature extraction as preprocessing ===")
print("""
# Extract features ONCE:
backbone.eval()
with torch.no_grad():
    # Remove the classifier
    feature_extractor = backbone.features
    features_tr = feature_extractor(X_tr_t)  # shape: (N, C, H', W')
    features_tr = nn.AdaptiveAvgPool2d(1)(features_tr).squeeze(-1).squeeze(-1)

# Then train a linear model on features_tr
linear = nn.Linear(features_tr.shape[1], 3)
# ... standard training loop
# Advantage: backbone forward pass runs ONCE instead of every batch/epoch
""")

print("=== Exercise 3 — Backbone comparison ===")
print("""
Approximate results (may vary):
  MobileNetV2 (frozen):  3.4M params total, ~500K trained   Fast, good accuracy
  Tiny custom CNN:       ~50K params                        Fast, lower accuracy
  ResNet18:              ~11M params                        Slower, possibly better
""")

print("=== Exercise 4 — Grad-CAM outline ===")
print("""
# Register hook on last conv layer
activations = {}
def hook(module, input, output):
    activations["last_conv"] = output

handle = backbone.features[-1].register_forward_hook(hook)

# Forward pass
output = backbone(image)
output[0, predicted_class].backward()

# Grad-CAM
grads = activations["last_conv"].grad.mean(dim=[2,3], keepdim=True)
cam = (grads * activations["last_conv"]).sum(dim=1).relu()
cam = F.interpolate(cam.unsqueeze(0), 96, mode="bilinear").squeeze()
""")

print("=== Exercise 5 — Domain shift ===")
print("""
# Test on inverted images
X_te_dark = 1 - X_te_original

# Solution: include dark-background images in training (domain adaptation)
# Or apply contrast/brightness augmentation during training to make model robust.
""")
