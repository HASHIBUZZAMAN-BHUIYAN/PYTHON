"""
cnn_feature_map_visualizer.py
==============================
What it does:
    Builds a small 3-conv-layer CNN (same architecture as medical_image_classifier.py),
    generates a single synthetic input image, registers PyTorch forward hooks on each
    Conv2d layer to capture intermediate feature maps, runs one forward pass, then saves
    a matplotlib grid showing the first 8 feature-map channels from each layer.

What it teaches:
    - PyTorch forward hooks for intermediate layer inspection
    - How spatial resolution and channel count change through a CNN
    - Visualizing what each conv layer "sees" / "activates on"
    - Understanding feature hierarchy: low-level edges -> higher-level patterns

RAM estimate  : ~150 MB peak
Time estimate : < 10 seconds on CPU
Real vs simulated: SIMULATED. Input image is a procedurally generated synthetic
    medical-style image (same generator as medical_image_classifier.py).
    No real images are used.
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

OUTPUT_DIR = "CNN/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 64
SEED     = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# -------------------------------------------------------------------------
# Reuse the same tiny CNN architecture from medical_image_classifier.py
# -------------------------------------------------------------------------
class MedCNN(nn.Module):
    """
    3-conv-layer CNN:
      Conv1: 1->8  ch, 3x3, ReLU, MaxPool2d(2)  -> (8,  32, 32)
      Conv2: 8->16 ch, 3x3, ReLU, MaxPool2d(2)  -> (16, 16, 16)
      Conv3: 16->16 ch, 3x3, ReLU, MaxPool2d(2) -> (16,  8,  8)
      FC: 1024 -> 64 -> 2
    """
    def __init__(self):
        super().__init__()
        # Name each layer explicitly so hooks can reference them by name
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        self.flatten    = nn.Flatten()
        self.fc1        = nn.Linear(16 * 8 * 8, 64)
        self.relu_fc    = nn.ReLU()
        self.dropout    = nn.Dropout(0.3)
        self.fc2        = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.dropout(self.relu_fc(self.fc1(self.flatten(x))))
        return self.fc2(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------------------------------------------------------
# Synthetic input image (medical-style, same generator style)
# -------------------------------------------------------------------------
def make_sample_image():
    """Generate one synthetic 64x64 'abnormal' image with a bright Gaussian blob."""
    rng = np.random.default_rng(SEED)
    base = np.full((IMG_SIZE, IMG_SIZE), 0.45, dtype=np.float32)
    # Blob at center-ish
    cx, cy = 36, 30
    sigma = 8.0
    x = np.arange(IMG_SIZE)
    y = np.arange(IMG_SIZE)
    xx, yy = np.meshgrid(x, y)
    blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)
    noise = rng.standard_normal((IMG_SIZE, IMG_SIZE)).astype(np.float32) * 0.04
    img = np.clip(base + 0.4 * blob + noise, 0.0, 1.0)
    return img


# -------------------------------------------------------------------------
# Forward hook registration
# -------------------------------------------------------------------------
def register_hooks(model):
    """
    Register a forward hook on each Conv2d layer.
    Hook captures the layer output (post-activation is not captured here;
    hooks fire AFTER each module's forward, so we capture conv output before ReLU).
    Returns a dict mapping layer_name -> list[tensor] and the hook handles.
    """
    feature_maps = {}
    handles = []

    def make_hook(name):
        def hook_fn(module, input, output):
            # output shape: (batch, channels, H, W)
            feature_maps[name] = output.detach().cpu()
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handle = module.register_forward_hook(make_hook(name))
            handles.append(handle)
            print(f"  Hook registered on: {name}  "
                  f"(in_ch={module.in_channels}, out_ch={module.out_channels}, "
                  f"kernel={module.kernel_size})")

    return feature_maps, handles


# -------------------------------------------------------------------------
# Feature map visualization
# -------------------------------------------------------------------------
def save_feature_maps(feature_maps, input_img, output_path):
    """
    Save a grid showing:
      - Top row: original input image (repeated across columns for alignment)
      - One row per conv layer: first 8 channels of that layer's feature map
    """
    layer_names = sorted(feature_maps.keys())   # conv1, conv2, conv3
    n_layers    = len(layer_names)
    n_channels  = 8   # show first 8 channels per layer

    # Layout: (n_layers + 1) rows, n_channels cols
    n_rows = n_layers + 1
    n_cols = n_channels

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.8))
    fig.suptitle("CNN Feature Map Visualization", fontsize=12, y=1.01)

    # Row 0: show input image in first column, blank rest
    axes[0, 0].imshow(input_img, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Input", fontsize=8)
    axes[0, 0].axis("off")
    for col in range(1, n_cols):
        axes[0, col].axis("off")

    print("\nFeature map details:")
    for row, layer_name in enumerate(layer_names, start=1):
        fmap = feature_maps[layer_name]   # (1, C, H, W)
        n_total_ch = fmap.shape[1]
        h, w = fmap.shape[2], fmap.shape[3]
        n_vis = min(n_channels, n_total_ch)
        print(f"  {layer_name:<10s}  output shape: (1, {n_total_ch}, {h}, {w})"
              f"  -- visualizing first {n_vis} filters")

        for col in range(n_cols):
            ax = axes[row, col]
            if col < n_vis:
                ch_data = fmap[0, col].numpy()   # (H, W)
                ax.imshow(ch_data, cmap="viridis")
                ax.set_title(f"{layer_name}\nch{col}", fontsize=6)
                ax.axis("off")
            else:
                ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=80, bbox_inches="tight")
    plt.close()
    print(f"\nFeature map grid saved -> {output_path}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("CNN Feature Map Visualizer")
    print("=" * 60)

    model = MedCNN()
    print(f"Model parameters: {count_params(model):,}")
    model.eval()  # set to eval mode (disables dropout for inference)

    print("\nRegistering forward hooks on Conv2d layers:")
    feature_maps, handles = register_hooks(model)

    # Generate sample input
    print("\nGenerating synthetic input image ...")
    sample_np = make_sample_image()
    # Shape (1, 1, H, W) for batch of 1
    x = torch.tensor(sample_np[np.newaxis, np.newaxis, :, :])

    # Forward pass - hooks fire automatically
    print("Running forward pass ...")
    with torch.no_grad():
        logits = model(x)
    pred_class = logits.argmax(dim=1).item()
    class_names = ["Normal", "Abnormal"]
    print(f"Model prediction: {class_names[pred_class]} "
          f"(logits: {logits[0].numpy().round(3).tolist()})")

    # Remove hooks after use
    for h in handles:
        h.remove()

    output_path = os.path.join(OUTPUT_DIR, "feature_maps.png")
    save_feature_maps(feature_maps, sample_np, output_path)

    print("\n[OK] cnn_feature_map_visualizer.py completed successfully.")


if __name__ == "__main__":
    main()
