"""
comms.py
Chunk-level partial reception simulation and parameter utilities.

Each recv_* function returns a 3-tuple:
    (reconstructed_flat, completeness, mask)

  reconstructed_flat : the parameter vector with missing chunks filled
                       (zero_fill or local_fill) — used by gossip / push-sum methods
  completeness       : c_ji = fraction of chunks received
  mask               : boolean array, length d. mask[k] = True iff coordinate k
                       was in a chunk that ARRIVED (= contains real sender data).
                       Used by Per-Coordinate Gossip (PCG) to aggregate only
                       over observed values.
"""
from __future__ import annotations
from typing import Callable, Tuple
import math
import numpy as np
import torch
import torch.nn as nn

from . import Topology
from .bandwidth import LinkBandwidth, apply_queue_loss


# ─────────────────────────────────────────────────────────────────────────────
# Packet sizing utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_packet_layout(
        d: int,
        packet_size_bytes: int = 1400,
        bytes_per_coord: int = 4,
) -> Tuple[int, int]:
    """
    Compute the chunk layout from packet size.

    Args:
        d: total number of coordinates in the flat parameter vector
        packet_size_bytes: UDP payload size per packet (default 1400 bytes)
        bytes_per_coord: 2 for fp16, 4 for fp32, 1 for int8

    Returns:
        (coords_per_packet, num_packets)
        - coords_per_packet: number of float values per packet
        - num_packets: total number of packets needed to transmit the model

    Standard MTU values:
        - 1400 bytes: safe UDP payload (accounts for headers)
        - 1500 bytes: Ethernet MTU
        - 9000 bytes: jumbo frames (data center networks)
    """
    coords_per_packet = max(1, packet_size_bytes // bytes_per_coord)
    num_packets = math.ceil(d / coords_per_packet)
    return coords_per_packet, num_packets


def to_flat(state_dict: dict, model: nn.Module = None) -> np.ndarray:
    """Flatten only LEARNABLE parameters (requires_grad=True).

    If model is provided, uses requires_grad to filter.
    Otherwise falls back to the float dtype filter (legacy behavior).
    """
    if model is not None:
        learnable_keys = {name for name, p in model.named_parameters()
                          if p.requires_grad}
        parts = [
            v.detach().cpu().float().numpy().ravel()
            for k, v in state_dict.items()
            if k in learnable_keys
        ]
    else:
        parts = [
            v.detach().cpu().float().numpy().ravel()
            for v in state_dict.values()
            if torch.is_tensor(v) and v.is_floating_point()
        ]

    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)


def from_flat(flat: np.ndarray, ref: dict, model: nn.Module = None) -> dict:
    """Inverse of to_flat with optional model parameter for requires_grad filter."""
    if model is not None:
        learnable_keys = {name for name, p in model.named_parameters()
                          if p.requires_grad}
    else:
        learnable_keys = None

    out, off = {}, 0
    for k, v in ref.items():
        if model is not None:
            include = k in learnable_keys
        else:
            include = torch.is_tensor(v) and v.is_floating_point()

        if include:
            n = v.numel()
            out[k] = (torch.from_numpy(flat[off:off + n].copy())
                      .reshape(v.shape).to(v.dtype))
            off += n
        else:
            out[k] = v.clone() if torch.is_tensor(v) else v

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Mask helper
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_mask_to_coord_mask(
        chunk_ok: np.ndarray,
        d: int,
        coords_per_packet: int,
) -> np.ndarray:
    """Expand packet-level arrival mask to per-coordinate mask."""
    coord_mask = np.zeros(d, dtype=bool)
    for k in range(len(chunk_ok)):
        s = k * coords_per_packet
        e = min(s + coords_per_packet, d)
        if chunk_ok[k]:
            coord_mask[s:e] = True
    return coord_mask


# ────────────────────────────────────────
# Channel models
# ────────────────────────────────────────

def recv_bernoulli(
        src_flat: np.ndarray,
        q_ji: float,
        packet_size_bytes: int,
        bytes_per_coord: int,
        local_flat: np.ndarray,
        mode: str,
        rng: np.random.Generator,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    i.i.d. Bernoulli packet loss.

    Args:
        src_flat: sender's parameter vector
        q_ji: per-packet reception probability
        packet_size_bytes: UDP payload size (typically 1400)
        bytes_per_coord: 2 for fp16, 4 for fp32
        local_flat: receiver's local parameters (for local_fill mode)
        mode: "zero_fill" or "local_fill"
        rng: numpy RNG

    Returns:
        (reconstructed_flat, completeness, coord_mask)
    """
    d = len(src_flat)
    coords_per_packet, num_packets = compute_packet_layout(
        d, packet_size_bytes, bytes_per_coord
    )

    packet_ok = rng.random(num_packets) < q_ji
    comp = float(packet_ok.sum()) / num_packets
    out = src_flat.copy()

    for k in range(num_packets):
        s = k * coords_per_packet
        e = min(s + coords_per_packet, d)
        if not packet_ok[k]:
            out[s:e] = 0.0 if mode == "zero_fill" else local_flat[s:e]

    coord_mask = _chunk_mask_to_coord_mask(packet_ok, d, coords_per_packet)
    return out, comp, coord_mask


def recv_gilbert_elliott(
        src_flat: np.ndarray,
        q_good: float,
        q_bad: float,
        p_gb: float,
        p_bg: float,
        packet_size_bytes: int,
        bytes_per_coord: int,
        local_flat: np.ndarray,
        mode: str,
        rng: np.random.Generator,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Bursty Gilbert-Elliott channel with packet-size-based chunking."""
    d = len(src_flat)
    coords_per_packet, num_packets = compute_packet_layout(
        d, packet_size_bytes, bytes_per_coord
    )

    pi_good = p_bg / (p_gb + p_bg)
    state = "good" if rng.random() < pi_good else "bad"
    packet_ok = np.zeros(num_packets, dtype=bool)

    for k in range(num_packets):
        q = q_good if state == "good" else q_bad
        packet_ok[k] = rng.random() < q
        if state == "good" and rng.random() < p_gb:
            state = "bad"
        elif state == "bad" and rng.random() < p_bg:
            state = "good"

    comp = float(packet_ok.sum()) / num_packets
    out = src_flat.copy()
    for k in range(num_packets):
        s = k * coords_per_packet
        e = min(s + coords_per_packet, d)
        if not packet_ok[k]:
            out[s:e] = 0.0 if mode == "zero_fill" else local_flat[s:e]

    coord_mask = _chunk_mask_to_coord_mask(packet_ok, d, coords_per_packet)
    return out, comp, coord_mask


def recv_rayleigh(
        src_flat: np.ndarray,
        snr_mean_lin: float,
        threshold_lin: float,
        packet_size_bytes: int,
        bytes_per_coord: int,
        local_flat: np.ndarray,
        mode: str,
        rng: np.random.Generator,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Rayleigh fading channel with packet-size-based chunking."""
    d = len(src_flat)
    coords_per_packet, num_packets = compute_packet_layout(
        d, packet_size_bytes, bytes_per_coord
    )

    snr = rng.exponential(snr_mean_lin, size=num_packets)
    packet_ok = snr > threshold_lin
    comp = float(packet_ok.sum()) / num_packets
    out = src_flat.copy()

    for k in range(num_packets):
        s = k * coords_per_packet
        e = min(s + coords_per_packet, d)
        if not packet_ok[k]:
            out[s:e] = 0.0 if mode == "zero_fill" else local_flat[s:e]

    coord_mask = _chunk_mask_to_coord_mask(packet_ok, d, coords_per_packet)
    return out, comp, coord_mask


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def make_recv_fn(channel_config: dict) -> Callable:
    """
    Returns a function with signature:
        fn(src, q, packet_size_bytes, bytes_per_coord, local, mode, rng)
        -> (flat, comp, mask)
    """
    model = channel_config.get("model", "bernoulli")

    if model == "bernoulli":
        def fn(src, q, packet_size_bytes, bytes_per_coord, local, mode, rng):
            return recv_bernoulli(
                src, q, packet_size_bytes, bytes_per_coord, local, mode, rng
            )

        return fn

    if model == "gilbert_elliott":
        q_good = float(channel_config.get("q_good", 0.95))
        q_bad = float(channel_config.get("q_bad", 0.30))
        p_gb = float(channel_config.get("p_gb", 0.05))
        p_bg = float(channel_config.get("p_bg", 0.10))

        def fn(src, q, packet_size_bytes, bytes_per_coord, local, mode, rng):
            return recv_gilbert_elliott(
                src, q_good, q_bad, p_gb, p_bg,
                packet_size_bytes, bytes_per_coord, local, mode, rng
            )

        return fn

    if model == "rayleigh":
        snr_db = float(channel_config.get("snr_mean_db", 10.0))
        thr_db = float(channel_config.get("threshold_db", 3.0))
        snr_lin = 10 ** (snr_db / 10)
        thr_lin = 10 ** (thr_db / 10)

        def fn(src, q, packet_size_bytes, bytes_per_coord, local, mode, rng):
            return recv_rayleigh(
                src, snr_lin, thr_lin,
                packet_size_bytes, bytes_per_coord, local, mode, rng
            )

        return fn

    raise ValueError(
        f"Unknown channel model: '{model}'. "
        f"Choose: bernoulli, gilbert_elliott, rayleigh"
    )
