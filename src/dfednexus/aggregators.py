"""
aggregators.py
All DFL aggregation methods including PCG and adapted baselines for comparison.
"""

from __future__ import annotations
from typing import Dict, Optional, Any, Tuple
import numpy as np

def _unpack_flat_comp(value: tuple):
    return value[0], value[1]


# ─────────────────────────────────────────────────────────────────────────────
# Gossip methods
# ─────────────────────────────────────────────────────────────────────────────

class DPSGD:
    """
    Vanilla Decentralized Parallel SGD (Lian et al., NeurIPS 2017).
    """

    def __init__(self):
        self.num_nodes = None
        self.node_id = None

    def reset(self, node_id, num_nodes, config):
        self.node_id = node_id
        self.num_nodes = num_nodes

    def aggregate(self, local, received, mixing_matrix=None, model_dict=None):
        if mixing_matrix is None:
            raise ValueError(
                "DPSGD requires a doubly-stochastic mixing_matrix. "
                "Use Metropolis-Hastings or similar topology-derived weights."
            )

        if not received:
            return local.copy()

        i = self.node_id
        w_self = mixing_matrix[i, i]

        agg = w_self * local.copy()

        for sender_id, msg in received.items():
            w_ij = mixing_matrix[i, sender_id]
            if w_ij <= 0:
                continue
            agg = agg + w_ij * msg[0] 

        return agg

    def get_state(self):
        return {}


class ZeroFill:
    """
    Zero-fill imputation aggregator for decentralized FL under packet loss.

    """

    def __init__(self):
        self.node_id = None
        self.num_nodes = None

    def reset(self, node_id, num_nodes, config):
        self.node_id = node_id
        self.num_nodes = num_nodes

    def aggregate(self, local, received, mixing_matrix=None, model_dict=None):

        if mixing_matrix is None:
            raise ValueError(
                "Zero-Fill requires a mixing_matrix argument. "
                "Compute Metropolis-Hastings or uniform W and pass it in."
            )

        if not received:
            return local.copy()

        i = self.node_id
        w_self = mixing_matrix[i, i]

        agg = w_self * local.copy()

        total_weight = w_self
        for sender_id, msg in received.items():
            z_filled = msg[0]  # already has zero-fill applied
            w_ij = mixing_matrix[i, sender_id]

            if w_ij <= 0:
                continue

            agg = agg + w_ij * z_filled
            total_weight += w_ij

        if abs(total_weight - 1.0) > 1e-6:
            agg = agg / total_weight

        return agg

    def get_state(self):
        return {}
    
class SoftDSGD:
    """
        Uniform gossip averaging with local-fill imputation.
        Aggregate per the Soft-DSGD consensus update rule.
    """

    def __init__(self):
        self.node_id = None
        self.num_nodes = None

    def reset(self, node_id, num_nodes, config):
        self.node_id = node_id
        self.num_nodes = num_nodes

    def aggregate(self, local, received, mixing_matrix=None, model_dict=None):

        if mixing_matrix is None:
            raise ValueError(
                "SoftDSGD requires a mixing_matrix argument. "
                "Compute Metropolis-Hastings or uniform W and pass it in."
            )

        if not received:
            return local.copy()

        i = self.node_id
        w_self = mixing_matrix[i, i]

        agg = w_self * local.copy()

        total_weight = w_self
        for sender_id, msg in received.items():
            z_filled = msg[0]  
            w_ij = mixing_matrix[i, sender_id]

            if w_ij <= 0:
                continue

            agg = agg + w_ij * z_filled
            total_weight += w_ij

        if abs(total_weight - 1.0) > 1e-6:
            agg = agg / total_weight

        return agg
    def get_state(self): return {}
    
class FedAvg:
    """
        Classical FedAvg aggregation.
        Drop if there is loss
    """

    def __init__(self):
        self.min_loss = None
        self.max_loss = None
        self.node_id = None
        self.num_nodes = None

    def reset(self, node_id, num_nodes, config):
        self.node_id = node_id
        self.num_nodes = num_nodes
        self.min_loss = config["loss_min"]
        self.max_loss = config["loss_max"]

    def aggregate(self, local, received, mixing_matrix=None, model_dict=None):

        if not received:
            return local.copy()

        if self.min_loss > 0.0:
            return local.copy()


        if mixing_matrix is None:
            raise ValueError(
                "Zero-Fill requires a mixing_matrix argument. "
                "Compute Metropolis-Hastings or uniform W and pass it in."
            )


        i = self.node_id
        w_self = mixing_matrix[i, i]
        
        agg = w_self * local.copy()

        total_weight = w_self
        for sender_id, msg in received.items():
            z_filled = msg[0] 
            w_ij = mixing_matrix[i, sender_id]

            if w_ij <= 0:
                continue

            agg = agg + w_ij * z_filled
            total_weight += w_ij

        if abs(total_weight - 1.0) > 1e-6:
            agg = agg / total_weight

        return agg
    def get_state(self): return {}


class Swift:
    """
    SWIFT: Rapid Decentralised FL via Wait-Free Model Communication.
    """

    def reset(self, node_id: int, num_nodes: int, config: dict):
        self.node_id = node_id
        raw = config.get("swift_local_weight", 0.5)
        self._alpha_fixed = float(raw) if raw is not None else None
        self._use_mh = bool(config.get("swift_use_mixing_matrix", False))

    def aggregate(self, local, received, mixing_matrix=None, model_dict=None):
        if not received:
            return local.copy()

        n_received = len(received)
        i = self.node_id

        alpha = (
            float(np.clip(self._alpha_fixed, 1e-6, 1.0 - 1e-6))
            if self._alpha_fixed is not None
            else 1.0 / (1 + n_received)
        )

        agg = alpha * local.astype(np.float32)

        if self._use_mh and mixing_matrix is not None:
            for sender_id, val in received.items():
                agg += float(mixing_matrix[i, sender_id]) * val[0].astype(np.float32)
        else:
            nbr_w = (1.0 - alpha) / n_received
            for val in received.values():
                agg += nbr_w * val[0].astype(np.float32)

        return agg

    def get_state(self) -> dict:
        return {
            "alpha": self._alpha_fixed if self._alpha_fixed else "adaptive",
            "use_mh": self._use_mh,
        }

class AdPSGD:
    def reset(self, node_id, num_nodes, config):
        self.rng = np.random.default_rng(node_id)
    def aggregate(self, local, received, mixing_matrix=None, model_dict=None):
        if not received: return local.copy()
        j = self.rng.choice(list(received.keys()))
        rv = _unpack_flat_comp(received[j])[0]
        return 0.5 * local + 0.5 * rv
    def get_state(self): return {}


class IPWGossip:
    """
    DFL-AA: IPW × AoI Adaptive Aggregation for Async DFL.

    Designed for asynchronous, directed DFL over lossy wireless links.
    Addresses two independent sources of update quality degradation:

      1. Spatial incompleteness (partial reception):
         Weights each neighbour by 1/q_hat_ij — the inverse of its
         estimated reception rate.  This is the Horvitz-Thompson
         estimator: unbiased correction for biased sampling.

      2. Temporal staleness (asynchrony):
         Weights each neighbour by exp(-AoI_ij / tau), where
         AoI_ij = (most recent gen_time in inbox) - gen_time_j.
         Relative AoI — no global clock needed.

    Combined weight:
        a_ij = (1 / q_hat_ij) * exp(-AoI_ij / tau)

    Aggregate (normalised weighted average):
        theta_i^new = (theta_i + sum_j a_ij * theta_j^rx) / (1 + sum_j a_ij)

    where theta_j^rx is the local-fill reconstructed parameter vector
    (missing chunks replaced with receiver's own values).

    Config keys
    ───────────
    dflaa_tau        : float   AoI decay constant in virtual-time units  (default 5.0)
    dflaa_beta       : float   EMA factor for q_hat update  (default 0.05)
    dflaa_c_min      : float   minimum completeness to use update  (default 0.1)
    dflaa_q_floor    : float   minimum q_hat to avoid 1/q explosion  (default 0.05)
    """

    def reset(self, node_id: int, num_nodes: int, config: dict):
        self.node_id = node_id
        self.tau = float(config.get("dflaa_tau", 5.0))
        self.beta = float(config.get("dflaa_beta", 0.05))
        self.c_min = float(config.get("dflaa_c_min", 0.10))
        self.q_floor = float(config.get("dflaa_q_floor", 0.05))
        self.q_hat: dict = {} 

    def aggregate(
            self,
            local: np.ndarray,
            received: dict,
            mixing_matrix=None,
            model_dict=None,
    ) -> np.ndarray:

        if not received:
            return local.copy()

        d = len(local)

        # ── Step 1: Compute relative AoI from gen_times in inbox ─────────────
        gen_times = [
            float(val[4]) if len(val) > 4 else 0.0
            for val in received.values()
        ]
        t_now = max(gen_times) if gen_times else 0.0

        # ── Step 2: Weighted aggregation ─────────────────────────────────────
        agg = local.astype(np.float32, copy=True)  
        Z = 1.0

        for sender_id, val in received.items():
            flat = val[0]
            completeness = float(val[1])
            gen_time = float(val[4]) if len(val) > 4 else 0.0

            # Skip very incomplete updates (likely unusable)
            if completeness < self.c_min:
                continue

            # ── Update q_hat via EMA of observed completeness ────────────────
            prev_q = self.q_hat.get(sender_id, completeness)
            self.q_hat[sender_id] = (
                    (1.0 - self.beta) * prev_q + self.beta * completeness
            )
            q_hat = max(self.q_hat[sender_id], self.q_floor)

            # ── IPW factor: 1/q_hat (Horvitz-Thompson correction) ────────────
            ipw = 1.0 / q_hat

            # ── AoI factor: exponential decay ────────────────────────────────
            aoi = max(0.0, t_now - gen_time)
            aoi_decay = float(np.exp(-aoi / self.tau))

            # ── Combined weight ───────────────────────────────────────────────
            a_ij = ipw * aoi_decay

            agg += a_ij * flat.astype(np.float32)
            Z += a_ij

        return agg / Z

    def get_state(self) -> dict:
        return {
            "q_hat_mean": float(np.mean(list(self.q_hat.values())))
            if self.q_hat else 0.0,
            "q_hat_min": float(min(self.q_hat.values()))
            if self.q_hat else 0.0,
            "n_tracked": len(self.q_hat),
            "tau": self.tau,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Push-sum methods (kept for ablation)
# ─────────────────────────────────────────────────────────────────────────────

class PSNaive:
    def reset(self, node_id, num_nodes, config):
        self.s: Optional[np.ndarray] = None
        self.w: float = 1.0
    def reencode(self, theta):
        if self.s is None:
            self.s = theta.copy(); self.w = 1.0
        else:
            self.s = theta * self.w
    def aggregate(self, local, received, mixing_matrix=None, model_dict=None):
        if self.s is None:
            self.s = local.copy(); self.w = 1.0
        if not received:
            return self.s / max(self.w, 1e-10)
        for val in received.values():
            rv = val[0]
            sender_w = val[2]
            self.s += rv
            self.w += sender_w
        if self.w < 1e-8:
            theta = self.s / max(self.w, 1e-12)
            self.s = theta.copy(); self.w = 1.0
            return theta
        return self.s / self.w
    def get_state(self):
        return {"w": round(float(self.w), 6) if self.w else 0.0}


class PSComp:
    def reset(self, node_id, num_nodes, config):
        self.s: Optional[np.ndarray] = None
        self.w: float = 1.0
    def reencode(self, theta):
        if self.s is None:
            self.s = theta.copy(); self.w = 1.0
        else:
            self.s = theta * self.w
    def aggregate(self, local, received, mixing_matrix=None, model_dict=None):
        if self.s is None:
            self.s = local.copy(); self.w = 1.0
        if not received:
            return self.s / max(self.w, 1e-10)
        for val in received.values():
            rv = val[0]
            comp = val[1]
            sender_w = val[2]
            self.s += rv
            self.w += comp * sender_w
        if self.w < 1e-8:
            theta = self.s / max(self.w, 1e-12)
            self.s = theta.copy(); self.w = 1.0
            return theta
        return self.s / self.w
    def get_state(self):
        return {"w": round(float(self.w), 6) if self.w else 0.0}



# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, type] = {
    "dpsgd":             DPSGD,
    "DPSGD":             DPSGD,
    "D-PSGD":            DPSGD,
    "d-psgd":            DPSGD,
    "gossip":            DPSGD,

    "zero-fill":         ZeroFill,
    "zerofill":          ZeroFill,
    "zero":              ZeroFill,
    "ZERO-FILL":         ZeroFill,
    "ZEROFILL":          ZeroFill,

    "softDSGD":          SoftDSGD,
    "soft_dsgd_uniform": SoftDSGD,
    "softdsgd-uniform":  SoftDSGD,
    "soft_dsgd_optimal": SoftDSGD,
    "softdsgd-optimal":  SoftDSGD,
    "softdsgd":          SoftDSGD,
    "soft-dsgd":         SoftDSGD,
    "soft-DSGD":         SoftDSGD,

    "swift":             Swift,
    "Swift":             Swift,
    "SWIFT":             Swift,

    "ad-psgd":           AdPSGD,
    "adpsgd":            AdPSGD,
    "ADPSGD":            AdPSGD,
    "AD-PSGD":           AdPSGD,

    "FedAvg":            FedAvg,
    "fedavg":            FedAvg,
    "FEDAVG":            FedAvg,

    "DFLAA":             IPWGossip,
    "dflaa":             IPWGossip,
    "ipw_gossip":        IPWGossip,
    "ipw-gossip":        IPWGossip,
    "ipwgossip":         IPWGossip,

    "push_sum":          PSComp,
    "pscomp":            PSComp,
    "ps_comp":           PSComp,
    "push_sum_naive":    PSNaive,
    "ps_naive":          PSNaive,

}


def make_aggregator(name: str):
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown aggregator '{name}'. Available: {sorted(_REGISTRY.keys())}")
    return cls()

def get_aggr_from_cls(cls_name):
    keys = [key for key, cls in _REGISTRY.items() if cls is cls_name]
    if not keys:
        raise ValueError(
            f"Unknown aggregator class '{cls_name}'. Available: {sorted(_REGISTRY.keys())}")

    return keys

def is_push_sum(agg) -> bool:
    return isinstance(agg, (PSComp, PSNaive))

def is_zerofill(agg) -> bool:
    return isinstance(agg, ZeroFill)

def list_methods() -> list:
    return sorted(set(v.__name__ for v in _REGISTRY.values()))