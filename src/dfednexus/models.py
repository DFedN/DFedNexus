"""
models.py
Neural network models for FedWave experiments.

SmallCNN  — fast model for sweeps and ablations (~500K params)
ResNet18  — full model for final paper results (~11M params)
MLP       — minimal model for theory validation
"""
from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class _DepthwiseSep(nn.Module):
    """
    Depthwise-separable convolution block: DW-Conv → PW-Conv → BN → ReLU.

    Replaces a standard k×k conv with ~8× fewer multiply-adds while
    preserving the receptive field. Optionally applies stride on the
    depthwise step for spatial downsampling.
    """

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False
        )
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))), inplace=True)


class _SEBlock(nn.Module):
    """
    Squeeze-and-Excite channel attention (Hu et al., CVPR 2018).

    Global average-pools each channel to a scalar, passes through a small
    bottleneck MLP, and rescales the feature map channel-wise with sigmoid
    gates. Adds <1% extra parameters for a consistent +1–2% accuracy gain.
    """

    def __init__(self, c: int, reduction: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, max(c // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(c // reduction, 4), c),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.se(x).view(x.size(0), -1, 1, 1)
        return x * w


class _ResBlock(nn.Module):
    """
    Residual block for EfficientSmallCNN.

    Layout:  DWSep(c→c) → DWSep(c→c) → SE → +skip → ReLU

    The skip connection is an identity (no projection needed — channel
    width is preserved within a stage). This fixes the vanishing-gradient
    problem that limited SmallCNN's convergence on deeper features.
    """

    def __init__(self, c: int):
        super().__init__()
        self.branch = nn.Sequential(
            _DepthwiseSep(c, c),
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.se = _SEBlock(c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.se(self.branch(x)) + x)


# ─────────────────────────────────────────────────────────────────────────────
# Main models
# ─────────────────────────────────────────────────────────────────────────────

class DeeperCNN(nn.Module):
    """
    Enhanced lightweight CNN. Drop-in replacement for SmallCNN.

    Architecture
    ------------
    Stem   : Conv2d(in_c → 32, 3×3) + BN + ReLU          [H×W preserved]
    Stage 1: DWSep(32→64, stride=2) + ResBlock(64)         [H/2 × W/2]
    Stage 2: DWSep(64→128, stride=2) + ResBlock(128)       [H/4 × W/4]
    Stage 3: DWSep(128→256, stride=2) + ResBlock(256)      [H/8 × W/8]
    Head   : GAP → Linear(256→256) → ReLU → Drop(0.2) → Linear(256→C)

    Parameters: ~319K (CIFAR-10 config)
    Compatible input sizes: 32×32 (CIFAR), 28×28 (MNIST family), larger.
    in_channels=1 supported for EMNIST/MNIST/FashionMNIST.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            _DepthwiseSep(32, 64, stride=2),
            _ResBlock(64),
        )
        self.stage2 = nn.Sequential(
            _DepthwiseSep(64, 128, stride=2),
            _ResBlock(128),
        )
        self.stage3 = nn.Sequential(
            _DepthwiseSep(128, 256, stride=2),
            _ResBlock(256),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────

class DeepCNN(nn.Module):
    """
    Fast CNN for FL sweep experiments.

    Design principle: wide + shallow + standard convs.
    GPU/MPS training throughput is the priority.
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()

        # Stage 1: 32x32 -> 16x16, standard conv, no skip needed (cheap)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # 32x32 -> 16x16
        )

        # Stage 2: 16x16 -> 8x8, wider, still standard conv
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # 16x16 -> 8x8
        )

        # Stage 3: 8x8 -> 4x4, single residual skip here (widest = most benefit)
        self.stage3_main = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.stage3_act  = nn.ReLU(inplace=True)
        self.pool3       = nn.MaxPool2d(2)            # 8x8 -> 4x4

        # Head: GAP -> classifier
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                  # 4x4 -> 1x1, MPS-safe
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)

        # Stage 3 with identity skip
        x = self.pool3(self.stage3_act(self.stage3_main(x) + x))

        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────

class SmallCNN(nn.Module):
    """
    Lightweight CNN for fast experiments.
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),          nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),          nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, num_classes: int = 10, in_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """
    ResNet-18 adapted for CIFAR 32x32 input.
    Use ONLY for final paper main results table.
    """
    from torchvision.models import resnet18
    m = resnet18(weights=None)
    m.conv1   = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(m.fc.in_features, num_classes)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Dataset → model defaults

_NUM_CLASSES = {
    "cifar10": 10, "cifar100": 100,
    "emnist": 47,  "mnist": 10, "fmnist": 10,
}
_IN_CHANNELS = {
    "cifar10": 3, "cifar100": 3,
    "emnist": 1,  "mnist": 1,   "fmnist": 1,
}
_MLP_DIM = {
    "emnist": 784, "mnist": 784, "fmnist": 784,
}


def get_model(name: str, dataset: str) -> nn.Module:
    nc = _NUM_CLASSES.get(dataset, 10)
    ch = _IN_CHANNELS.get(dataset, 3)
    if name == "resnet18": return resnet18_cifar(nc)
    if name == "smallcnn": return SmallCNN(nc, ch)
    if name == "deepcnn": return DeepCNN(nc, ch)
    if name == "deepercnn": return DeeperCNN(nc, ch)
    if name == "mlp":      return MLP(nc, _MLP_DIM.get(dataset, 32 * 32 * ch))
    raise ValueError(f"Unknown model: '{name}'. Choose: resnet18, smallcnn, mlp")