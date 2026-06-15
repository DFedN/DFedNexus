"""
node.py
Lightweight in-process DFL node.

Inbox format (4-tuple):
    inbox[sender_id] = (flat, completeness, sender_alpha, coord_mask)

  flat         : reconstructed parameter vector (d,)
  completeness : c_ji = fraction of chunks received
  sender_alpha : 1 / (sender_out_degree + 1)
                 (= sender's mixing weight, used by push-sum methods)
  coord_mask   : boolean array (d,), True where coord was in an arrived chunk
                 (used by Per-Coordinate Gossip)

Aggregators extract whichever fields they need; older aggregators that
expect 3-tuples or 2-tuples still work because they index by position.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .communication import to_flat, from_flat
from .aggregators import make_aggregator, get_aggr_from_cls


class Node:
    def __init__(
        self,
        node_id:    int,
        model:      nn.Module,
        loader:     DataLoader,
        aggregator,
        device:     torch.device,
        config:     dict,
    ):
        self.i      = node_id
        self.model  = model.to(device)
        self.loader = loader
        self.agg    = aggregator
        self.device = device
        self.cfg    = config
        self._it    = iter(loader)

        ds = config.get("dataset", "cifar10")
        lr = float(config.get("lr", 0.01))
        wd = float(config.get("weight_decay", 5e-4))

        if ds in ("cifar10", "cifar100"):
            self.opt = optim.SGD(
                model.parameters(), lr=lr, momentum=0.9,
                weight_decay=wd, nesterov=True)
        else:
            self.opt = optim.Adam(
                model.parameters(), lr=lr, weight_decay=wd)

        self.crit        = nn.CrossEntropyLoss()
        self.local_round = 0
        self._last_loss  = 0.0

        # inbox[sender_id] = (flat, completeness, sender_alpha, coord_mask)     --> message inbox from each neighbor
        self.inbox: Dict[int, Tuple[np.ndarray, float, float, np.ndarray, float]] = {}

    def train_step(self, num_steps: int = 1):
        self.model.train()
        total = 0.0
        for _ in range(num_steps):
            try:
                x, y = next(self._it)
            except StopIteration:
                self._it = iter(self.loader)
                x, y    = next(self._it)
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad(set_to_none=True)
            loss = self.crit(self.model(x), y)
            loss.backward()
            if self.cfg.get("dataset", "") in ("cifar10", "cifar100"):
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            total += loss.item()
        self.local_round += 1
        self._last_loss   = total / num_steps

    # only belongs to push sum variants: currently not using
    def reencode(self):
        if hasattr(self.agg, "reencode"):
            self.agg.reencode(self.flat())

    # only belongs to push sum variants: currently not using
    def get_push_sum_weight(self) -> Optional[float]:
        return float(self.agg.w) if hasattr(self.agg, "w") else None

    def flat(self) -> np.ndarray:
        return to_flat(self.model.state_dict())

    def set_flat(self, f: np.ndarray):
        self.model.load_state_dict(
            from_flat(f, self.model.state_dict()), strict=True)

    def receive(
            self,
            sender: int,
            flat: np.ndarray,
            completeness: float,
            sender_alpha: float,
            coord_mask: np.ndarray,
            gen_time: float = 0.0,
    ):
        """Store latest message from sender as 5-tuple.

        inbox[sender] = (flat, completeness, sender_alpha, coord_mask, gen_time)

          gen_time : virtual clock time when the sender broadcast this message.
                     Used by DFLAA aggregator to compute Age-of-Information (AoI).
                     Defaults to 0.0 (backward-compatible with all other aggregators).
        """
        self.inbox[sender] = (flat, completeness, sender_alpha, coord_mask, gen_time)

    # aggregation might expect to have pre-computed mixing matrix
    def aggregate(self, mixing_matrix=None):
        if not self.inbox:
            return

        
        new_flat = self.agg.aggregate(self.flat(), self.inbox, mixing_matrix,
                                      self.model.state_dict()) 
        self.set_flat(new_flat)

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        correct = total = 0
        loss_sum = 0.0
        for x, y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            out   = self.model(x)
            loss_sum += self.crit(out, y).item() * y.size(0)
            correct  += (out.argmax(1) == y).sum().item()
            total    += y.size(0)
        return correct / total, loss_sum / total