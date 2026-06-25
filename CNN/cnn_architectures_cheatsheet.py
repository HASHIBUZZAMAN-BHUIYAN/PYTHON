# CNN Reference — Architecture Cheat-Sheet
# CPU-friendly examples. All run on 32×32 or 64×64 inputs.
# ~60 MB RAM, <3s on CPU

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── 1. LeNet-style (2-conv) ──────────────────────────────────────────────────
class LeNetStyle(nn.Module):
    """Classic 2-conv CNN. Input: (B,1,32,32). For grayscale."""
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6,  5), nn.Tanh(), nn.AvgPool2d(2,2),
            nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2,2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120), nn.Tanh(),
            nn.Linear(120, 84),     nn.Tanh(),
            nn.Linear(84, n_classes),
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))

# ─── 2. VGG-mini (3-conv blocks) ────────────────────────────────────────────
class VGGMini(nn.Module):
    """3 VGG-style conv blocks. Input: (B,3,64,64)."""
    def __init__(self, n_classes=10):
        super().__init__()
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
                nn.MaxPool2d(2,2),
            )
        self.features = nn.Sequential(block(3,32), block(32,64), block(64,128))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2,2)),
            nn.Flatten(),
            nn.Linear(128*4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))

# ─── 3. ResNet Residual Block ─────────────────────────────────────────────────
class ResBlock(nn.Module):
    """Standard ResNet residual block with optional downsampling."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.skip  = nn.Sequential() if (in_ch==out_ch and stride==1) else \
                     nn.Sequential(nn.Conv2d(in_ch,out_ch,1,stride=stride,bias=False),
                                   nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))

class TinyResNet(nn.Module):
    """Tiny ResNet for 32×32 images (e.g. CIFAR-style)."""
    def __init__(self, n_classes=10):
        super().__init__()
        self.stem  = nn.Sequential(nn.Conv2d(3,32,3,padding=1,bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.layer1= nn.Sequential(ResBlock(32,32), ResBlock(32,32))
        self.layer2= nn.Sequential(ResBlock(32,64,stride=2), ResBlock(64,64))
        self.layer3= nn.Sequential(ResBlock(64,128,stride=2), ResBlock(128,128))
        self.head  = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
                                   nn.Linear(128, n_classes))
    def forward(self, x):
        x=self.stem(x); x=self.layer1(x); x=self.layer2(x); x=self.layer3(x)
        return self.head(x)

# ─── 4. Depthwise Separable Conv (MobileNet-style) ────────────────────────────
class DepthwiseSeparable(nn.Module):
    """Depthwise + pointwise conv. ~8–9x fewer parameters than standard conv."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(),
        )
    def forward(self, x): return self.pw(self.dw(x))

class MobileNetMini(nn.Module):
    """MobileNet-style network. Input: (B,3,64,64)."""
    def __init__(self, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,stride=2,padding=1,bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            DepthwiseSeparable(32,64),  DepthwiseSeparable(64,128,stride=2),
            DepthwiseSeparable(128,128),DepthwiseSeparable(128,256,stride=2),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
            nn.Dropout(0.2), nn.Linear(256, n_classes),
        )
    def forward(self, x): return self.net(x)

# ─── 5. Transfer Learning Skeleton (torchvision) ──────────────────────────────
def make_transfer_model(n_classes, backbone="mobilenet_v2", freeze=True):
    """
    Returns a transfer model with frozen backbone.
    Use backbone="resnet18" or "mobilenet_v2".
    Tested with CPU — MobileNetV2 needs input ≥ 96×96.
    """
    from torchvision import models
    if backbone == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        if freeze:
            for p in model.parameters(): p.requires_grad = False
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_f, n_classes))
    elif backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if freeze:
            for p in model.parameters(): p.requires_grad = False
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, n_classes)
    return model

# ─── PARAMETER COUNT UTILITY ──────────────────────────────────────────────────
def count_params(model):
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# ─── DEMO: compare sizes ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"{'Architecture':<22} {'Total params':>14} {'Trainable':>12}")
    print("-"*50)
    x32 = torch.randn(2,1,32,32)
    x64_rgb = torch.randn(2,3,64,64)
    for name, model, x in [
        ("LeNetStyle",    LeNetStyle(10),    x32),
        ("VGGMini",       VGGMini(10),       x64_rgb),
        ("TinyResNet",    TinyResNet(10),     x64_rgb),
        ("MobileNetMini", MobileNetMini(10),  x64_rgb),
    ]:
        out = model(x)
        tot, train = count_params(model)
        print(f"  {name:<20} {tot:>14,} {train:>12,}  out={out.shape}")

    print("\n--- Transfer model (MobileNetV2, frozen backbone) ---")
    xfer = make_transfer_model(5, freeze=True)
    tot, train = count_params(xfer)
    print(f"  Total={tot:,}  Trainable={train:,}  ({100*train/tot:.2f}%)")
