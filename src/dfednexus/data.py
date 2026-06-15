"""
data.py
Dataset loading and Dirichlet non-IID partitioning.
"""
from __future__ import annotations
import os
from typing import List, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset


def load_dataset(name: str, root: str = "/tmp/data") -> Tuple[Dataset, Dataset]:
    import torchvision
    import torchvision.transforms as T

    os.makedirs(root, exist_ok=True)
    name = name.lower()

    if name == "cifar10":
        tr = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        return (torchvision.datasets.CIFAR10(root, True,  download=True, transform=tr),
                torchvision.datasets.CIFAR10(root, False, download=True, transform=tr))

    if name == "cifar100":
        tr = T.Compose([
            T.ToTensor(),
            T.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])
        return (torchvision.datasets.CIFAR100(root, True,  download=True, transform=tr),
                torchvision.datasets.CIFAR100(root, False, download=True, transform=tr))

    if name == "emnist":
        tr = T.Compose([T.ToTensor(), T.Normalize((0.1736,), (0.3248,))])
        return (torchvision.datasets.EMNIST(root, "balanced", train=True,  download=True, transform=tr),
                torchvision.datasets.EMNIST(root, "balanced", train=False, download=True, transform=tr))

    cls = (torchvision.datasets.FashionMNIST if name == "fmnist"
           else torchvision.datasets.MNIST)
    tr  = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    return (cls(root, True,  download=True, transform=tr),
            cls(root, False, download=True, transform=tr))


def dirichlet_partition(
    dataset:   Dataset,
    num_nodes: int,
    alpha:     float,
    seed:      int = 42,
) -> List[List[int]]:
    """
    Partition dataset using Dirichlet(alpha) distribution.

    alpha → 0  : extreme non-IID (each node has ~1 class)
    alpha = 0.1: high heterogeneity (typical hard FL benchmark)
    alpha = 1.0: moderate heterogeneity
    alpha → ∞  : IID (uniform distribution)
    """
    rng    = np.random.default_rng(seed)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    nc     = int(labels.max()) + 1

    class_idx = [np.where(labels == c)[0] for c in range(nc)]
    parts: List[List[int]] = [[] for _ in range(num_nodes)]

    for c_idx in class_idx:
        props  = rng.dirichlet(np.ones(num_nodes) * alpha)
        splits = (np.cumsum(props) * len(c_idx)).astype(int)
        for i, chunk in enumerate(np.split(c_idx, splits[:-1])):
            parts[i].extend(chunk.tolist())

    # Every node needs at least one sample
    for i in range(num_nodes):
        if not parts[i]:
            parts[i] = [int(rng.integers(0, len(dataset)))]

    return parts


def make_loader(
    dataset:    Dataset,
    indices:    List[int],
    batch_size: int  = 64,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )


def partition_summary(dataset: Dataset, parts: List[List[int]]) -> List[dict]:
    """Per-node statistics for logging."""
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    nc     = int(labels.max()) + 1
    info   = []
    for i, idx in enumerate(parts):
        arr    = np.array(idx)
        counts = np.bincount(labels[arr], minlength=nc).tolist()
        info.append({
            "node_id":       i,
            "num_samples":   len(idx),
            "class_counts":  counts,
            "dominant_class":int(np.argmax(counts)),
            "dominant_pct":  round(100 * max(counts) / max(len(idx), 1), 1),
        })
    return info