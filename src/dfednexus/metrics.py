"""
metrics.py
All metrics for DFL experiments.

Paper metrics
─────────────
1. Test accuracy (main result table)
2. Consensus distance Δ(t) — validates Theorem 2
3. Push-sum weight w_i    — validates Proposition 2 (weight drain)
4. Approx gradient norm²  — validates Theorem 3 convergence rate
5. Rounds to target accuracy — communication efficiency
6. Mean completeness c̄    — channel condition monitoring
7. AUC of accuracy curve   — summary metric
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def consensus_distance(flats: List[np.ndarray]) -> float:
    """
    Δ(t) = (1/n) Σ_i ||θ̂_i - θ̃||²
    θ̃ = (1/n) Σ_i θ̂_i  (network average)

    Validates Theorem 2: should contract geometrically under PS-Comp.
    Under biased methods: converges to a non-zero floor (irreducible bias).
    """
    if len(flats) < 2:
        return 0.0
    mat  = np.stack(flats, axis=0)
    mean = mat.mean(axis=0)
    return float(np.mean(np.sum((mat - mean) ** 2, axis=1)))


def mean_weight(weights: List[float]) -> float:
    """
    Average push-sum weight w̄ = (1/n) Σ w_i.
    Should stay ≈ 1.0 for PS-Comp.
    Drifts toward 0 for PS-naive (weight drain, Proposition 2).
    """
    return float(np.mean(weights)) if weights else 0.0


def approx_grad_norm_sq(
    model:       nn.Module,
    loader:      DataLoader,
    device:      torch.device,
    num_batches: int = 5,
) -> float:
    """
    Approximate ||∇F(θ̃)||² at the network-average model.
    Validates Theorem 3: should decay as O(1/√nT).
    Expensive — call every 5 eval intervals, not every round.
    """
    model.train()
    crit  = nn.CrossEntropyLoss()
    total = 0.0
    count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.detach().norm().item() ** 2
        count += 1
        if count >= num_batches:
            break
    return total / max(1, count)


def rounds_to_target(
    accuracy_log: List[float],
    rounds_log:   List[int],
    target:       float,
) -> Optional[int]:
    """First round at which mean accuracy ≥ target (%). None if not reached."""
    for r, a in zip(rounds_log, accuracy_log):
        if a >= target:
            return r
    return None


def compute_auc(accuracy_log: List[float], rounds_log: List[int]) -> float:
    """Area under accuracy-vs-rounds curve (trapezoidal rule)."""
    if len(accuracy_log) < 2:
        return 0.0

    try:
        return float(np.trapezoid(accuracy_log, rounds_log))
    except AttributeError:
        return float(np.trapz(accuracy_log, rounds_log))


# ─────────────────────────────────────────────────────────────────────────────
# AoI metrics  (require gen_time as 5th element in inbox tuples)
# ─────────────────────────────────────────────────────────────────────────────

def mean_inbox_aoi(nodes: list, t_now: float) -> float:
    """
    Mean Age-of-Information across all nodes' inboxes.

        AoI_ij = t_now − gen_time_j   (how old is j's model in i's inbox)

    Returns 0.0 if all inboxes are empty or have no gen_time field.
    """
    aois = []
    for nd in nodes:
        for val in nd.inbox.values():
            gen_time = float(val[4]) if len(val) > 4 else 0.0
            aois.append(t_now - gen_time)
    return float(np.mean(aois)) if aois else 0.0


def max_inbox_aoi(nodes: list, t_now: float) -> float:
    """Maximum AoI across all nodes' inboxes."""
    aois = []
    for nd in nodes:
        for val in nd.inbox.values():
            gen_time = float(val[4]) if len(val) > 4 else 0.0
            aois.append(t_now - gen_time)
    return float(np.max(aois)) if aois else 0.0


def update_reject_rate(nodes: list, c_min: float) -> float:
    """
    Fraction of inbox messages currently below the c_min threshold.
    Measures how aggressively the completeness filter is rejecting updates.
    """
    total = 0
    rejected = 0
    for nd in nodes:
        for val in nd.inbox.values():
            comp = float(val[1])
            total += 1
            if comp < c_min:
                rejected += 1
    return float(rejected / total) if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────────────────────────────────────

class MetricsTracker:
    """Accumulates all metrics across rounds for one method run."""

    def __init__(self):
        self.rounds:   List[int]   = []
        self.accuracy: List[float] = []
        self.loss:     List[float] = []
        self.cons_dist:List[float] = []
        self.mean_w:   List[float] = []
        self.mean_comp:List[float] = []
        self.grad_norm:List[float] = []
        self.time:     List[float] = []
        self.comp_at_tx: List[float] = []
        self.comp_at_rx: List[float] = []
        self.transmission_duration: List[float] = []

        self.mean_aoi: List[float] = []
        self.max_aoi: List[float] = []
        self.reject_rate: List[float] = []

    def record(
        self,
        rnd:       int,
        accuracy:  float,
        loss:      float,
        elapsed:   float,
        comp_tx:   float,
        comp_rx:   float,
        transmission_duration: float,
        cons_dist: Optional[float] = None,
        grad_norm: Optional[float] = None,
        mean_w:    Optional[float] = None,
        mean_comp: Optional[float] = None,
        mean_aoi: Optional[float] = None,
        max_aoi: Optional[float] = None,
        reject_rate: Optional[float] = None,
    ):
        self.rounds.append(rnd)
        self.accuracy.append(accuracy)
        self.loss.append(loss)
        self.time.append(elapsed)
        self.comp_at_tx.append(comp_tx)
        self.comp_at_rx.append(comp_rx)
        self.transmission_duration.append(transmission_duration)
        if cons_dist  is not None: self.cons_dist.append(cons_dist)
        if grad_norm  is not None: self.grad_norm.append(grad_norm)
        if mean_w     is not None: self.mean_w.append(mean_w)
        if mean_comp  is not None: self.mean_comp.append(mean_comp)
        if mean_aoi is not None: self.mean_aoi.append(mean_aoi)
        if max_aoi is not None: self.max_aoi.append(max_aoi)
        if reject_rate is not None: self.reject_rate.append(reject_rate)

    def summary(self) -> dict:
        return {
            "final_accuracy": self.accuracy[-1]   if self.accuracy   else 0.0,
            "max_accuracy":   max(self.accuracy)   if self.accuracy   else 0.0,
            "min_loss":       min(self.loss)        if self.loss       else 0.0,
            "final_cons_dist":self.cons_dist[-1]   if self.cons_dist  else None,
            "final_mean_w":   self.mean_w[-1]      if self.mean_w     else None,
            "total_time_s":   self.time[-1]         if self.time       else 0.0,
            "auc":            compute_auc(self.accuracy, self.rounds),
            "average_time_at_tx":  np.mean(self.comp_at_tx),
            "average_time_at_rx": np.mean(self.comp_at_rx),
            "average_time_at_duration": np.mean(self.transmission_duration),
            # round-based (used in sync mode and legacy async round mode)
            "rounds_to_50":   rounds_to_target(self.accuracy, self.rounds, 50.0),
            "rounds_to_60":   rounds_to_target(self.accuracy, self.rounds, 60.0),
            "rounds_to_70":   rounds_to_target(self.accuracy, self.rounds, 70.0),

            # time-based aliases (same values; labelled differently for clarity)
            # In time mode self.rounds contains virtual seconds, so these are
            # "virtual seconds to X% accuracy".
            "time_to_50": rounds_to_target(self.accuracy, self.rounds, 50.0),
            "time_to_60": rounds_to_target(self.accuracy, self.rounds, 60.0),
            "time_to_70": rounds_to_target(self.accuracy, self.rounds, 70.0),

            # AoI
            "final_mean_aoi": self.mean_aoi[-1] if self.mean_aoi else None,
            "max_aoi_ever": max(self.max_aoi) if self.max_aoi else None,
            "mean_reject_rate": float(np.mean(self.reject_rate)) if self.reject_rate else None,
        }

    def to_dict(self) -> dict:
        d = {
            "rounds":    self.rounds,
            "accuracy":  self.accuracy,
            "loss":      self.loss,
            "cons_dist": self.cons_dist,
            "mean_w":    self.mean_w,
            "mean_comp": self.mean_comp,
            "grad_norm": self.grad_norm,
            "time":      self.time,
            "comp_at_tx": self.comp_at_tx,
            "comp_at_rx": self.comp_at_rx,
            "transmission_duration": self.transmission_duration,

            "mean_aoi": self.mean_aoi,
            "max_aoi": self.max_aoi,
            "reject_rate": self.reject_rate,
        }
        d.update(self.summary())
        return d