"""
bandwidth.py
Realistic wireless bandwidth modeling for DFL sync simulation.

Three components:
  1. LinkBandwidth         — per-link Mbps capacity with optional time variation
  2. BandwidthMatrix       — manages all per-link bandwidth across the topology
  3. MM1KQueue             — finite-buffer queue producing utilization-induced loss

Designed to compose with existing comms.py recv_* functions: this module
computes transmission delays and additional queue-induced loss, which can
be applied on top of channel-induced loss from recv_bernoulli /
recv_gilbert_elliott / recv_rayleigh.
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Callable


# ─────────────────────────────────────────────────────────────────────────────
# Per-link bandwidth parameterization
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LinkBandwidth:
    """Bandwidth parameters for one directed link (i → j).

    Attributes:
        capacity_mbps: nominal link capacity in Mbps
        rtt_ms: round-trip time for ACK/feedback (relevant for protocol overhead)
        time_varying: if True, capacity fluctuates per round (mobile fading)
        variation_amplitude: fraction of capacity (0.0 - 1.0) by which
            time-varying bandwidth oscillates
        queue_buffer_size: K parameter for M/M/1/K queueing model (0 = infinite)
        queue_utilization: rho = lambda/mu, fixed at config time
    """
    capacity_mbps: float = 10.0
    rtt_ms: float = 20.0
    time_varying: bool = False
    variation_amplitude: float = 0.3
    queue_buffer_size: int = 0
    queue_utilization: float = 0.8

    def current_capacity_mbps(self, round_idx: int, rng: np.random.Generator) -> float:
        """Compute effective capacity at this round.

        Stable links return capacity unchanged. Time-varying links oscillate
        sinusoidally to model slow fading (e.g., mobility-induced channel changes).
        """
        if not self.time_varying:
            return self.capacity_mbps
        phase = (round_idx % 100) / 100.0
        modulation = 1.0 + self.variation_amplitude * math.sin(2 * math.pi * phase)
        return max(0.1, self.capacity_mbps * modulation)


# ─────────────────────────────────────────────────────────────────────────────
# Heterogeneous bandwidth matrix
# ─────────────────────────────────────────────────────────────────────────────

class BandwidthMatrix:
    """Holds LinkBandwidth for every active edge in the topology.

    Supports homogeneous (same params everywhere) and heterogeneous
    (sampled from distributions) configurations. Use in conjunction with
    your existing Topology object.
    """

    def __init__(
            self,
            edges: list,  
            config: dict,
            rng: np.random.Generator,
    ):
        """
        config dict keys:
            mode: "homogeneous" | "heterogeneous" | "distance"
            capacity_mbps_mean: float
            capacity_mbps_std: float (for heterogeneous mode)
            capacity_mbps_min, capacity_mbps_max: float (range bounds)
            rtt_ms_mean: float
            time_varying: bool
            queue_buffer_size: int
            queue_utilization: float
            positions: optional dict {node_id: (x, y)} for distance mode
        """
        self.edges = edges
        self.config = config
        self.rng = rng
        self.links: Dict[Tuple[int, int], LinkBandwidth] = {}
        self._populate()

    def _populate(self):
        mode = self.config.get("mode", "homogeneous")
        cap_mean = float(self.config.get("capacity_mbps_mean", 10.0))
        cap_std = float(self.config.get("capacity_mbps_std", 2.0))
        cap_min = float(self.config.get("capacity_mbps_min", 1.0))
        cap_max = float(self.config.get("capacity_mbps_max", 100.0))
        rtt_mean = float(self.config.get("rtt_ms_mean", 20.0))
        time_varying = bool(self.config.get("time_varying", False))
        buffer_size = int(self.config.get("queue_buffer_size", 7))
        utilization = float(self.config.get("queue_utilization", 0.8))

        if mode == "homogeneous":
            for (src, dst) in self.edges:
                self.links[(src, dst)] = LinkBandwidth(
                    capacity_mbps=cap_mean,
                    rtt_ms=rtt_mean,
                    time_varying=time_varying,
                    queue_buffer_size=buffer_size,
                    queue_utilization=utilization,
                )

        elif mode == "heterogeneous":
            # Each link gets a random capacity sampled from truncated normal
            for (src, dst) in self.edges:
                cap = float(np.clip(
                    self.rng.normal(cap_mean, cap_std),
                    cap_min, cap_max
                ))
                self.links[(src, dst)] = LinkBandwidth(
                    capacity_mbps=cap,
                    rtt_ms=rtt_mean,
                    time_varying=time_varying,
                    queue_buffer_size=buffer_size,
                    queue_utilization=utilization,
                )

        elif mode == "distance":
            positions = self.config.get("positions", {})
            if not positions:
                raise ValueError("distance mode requires 'positions' in config")
            ref_distance = 100.0
            path_loss_exponent = 3.0
            for (src, dst) in self.edges:
                pos_src = positions[src]
                pos_dst = positions[dst]
                dist = math.sqrt(
                    (pos_src[0] - pos_dst[0]) ** 2
                    + (pos_src[1] - pos_dst[1]) ** 2
                )
                dist = max(1.0, dist) 
                ratio = (ref_distance / dist) ** path_loss_exponent
                cap = float(np.clip(cap_mean * ratio, cap_min, cap_max))
                self.links[(src, dst)] = LinkBandwidth(
                    capacity_mbps=cap,
                    rtt_ms=rtt_mean,
                    time_varying=time_varying,
                    queue_buffer_size=buffer_size,
                    queue_utilization=utilization,
                )

        else:
            raise ValueError(f"Unknown bandwidth mode: {mode}")

    def get_link(self, src: int, dst: int) -> LinkBandwidth:
        if (src, dst) not in self.links:
            raise KeyError(f"No link ({src}, {dst}) in bandwidth matrix")
        return self.links[(src, dst)]

    def summary(self) -> dict:
        """For logging: capacity distribution across links."""
        caps = [link.capacity_mbps for link in self.links.values()]
        return {
            "n_links": len(self.links),
            "capacity_mean": float(np.mean(caps)),
            "capacity_std": float(np.std(caps)),
            "capacity_min": float(np.min(caps)),
            "capacity_max": float(np.max(caps)),
            "capacity_median": float(np.median(caps)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Transmission delay computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_transmission_delay(
        payload_bytes: int,
        link: LinkBandwidth,
        round_idx: int,
        rng: np.random.Generator,
        protocol_overhead_factor: float = 1.05,  # 5% for headers
) -> float:
    """Compute total transmission delay for a payload over a link.

    Returns delay in seconds, accounting for:
      - Raw transmission time: bytes / bandwidth
      - One-way propagation: rtt/2
      - Protocol overhead: headers, packet framing

    Does NOT include retransmission delay (that's a higher-level concern).
    """
    capacity_bps = link.current_capacity_mbps(round_idx, rng) * 1e6
    effective_payload_bytes = payload_bytes * protocol_overhead_factor
    transmission_time = (effective_payload_bytes * 8) / capacity_bps
    propagation_time = (link.rtt_ms / 2.0) / 1000.0
    return transmission_time + propagation_time


# ─────────────────────────────────────────────────────────────────────────────
# M/M/1/K queue-induced loss TODO: Not used in the current version of simulation
# ─────────────────────────────────────────────────────────────────────────────

def mm1k_loss_probability(rho: float, K: int) -> float:
    """Compute M/M/1/K blocking probability analytically.

    Args:
        rho: traffic utilization factor (lambda / mu)
        K: total capacity (queue length + 1)

    Returns:
        probability that an arriving packet finds the system full and is dropped
    """
    if K <= 0:
        return 0.0  # infinite buffer, no queue loss
    if abs(rho - 1.0) < 1e-9:
        return 1.0 / (K + 1)
    return (1 - rho) * (rho ** K) / (1 - rho ** (K + 1))


def apply_queue_loss(
        base_completeness: float,
        link: LinkBandwidth,
) -> float:
    """Apply queue-induced loss on top of channel-induced loss.

    The two loss sources compose multiplicatively: a chunk survives if
    it neither gets dropped by the channel nor by the queue.

    Returns adjusted completeness in [0, base_completeness].
    """
    if link.queue_buffer_size <= 0:
        return base_completeness  # infinite buffer
    queue_loss = mm1k_loss_probability(
        link.queue_utilization, link.queue_buffer_size
    )
    survival_prob = 1.0 - queue_loss
    return base_completeness * survival_prob


# ─────────────────────────────────────────────────────────────────────────────
# Bandwidth metrics tracker (for logging / paper figures)
# ─────────────────────────────────────────────────────────────────────────────

class BandwidthMetrics:
    """Tracks bytes sent, delays incurred, and utilization per link per round.

    Use this to populate the bandwidth-utilization figures in your paper:
      - Bandwidth efficiency = useful_bytes / wire_bytes
      - Per-link utilization distribution
      - Aggregate network load
      - Channel airtime
    """

    def __init__(self):
        self.per_link_bytes_sent: Dict[Tuple[int, int], list] = {}
        self.per_link_useful_bytes: Dict[Tuple[int, int], list] = {}
        self.per_link_delays: Dict[Tuple[int, int], list] = {}
        self.per_round_max_delay: list = []
        self.per_round_total_wire_bytes: list = []

    def record(
            self,
            src: int, dst: int,
            useful_payload_bytes: int,
            wire_bytes: int,
            delay_seconds: float,
    ):
        key = (src, dst)
        self.per_link_bytes_sent.setdefault(key, []).append(wire_bytes)
        self.per_link_useful_bytes.setdefault(key, []).append(useful_payload_bytes)
        self.per_link_delays.setdefault(key, []).append(delay_seconds)

    def end_round(self, round_delays: list, round_wire_bytes: list):
        self.per_round_max_delay.append(
            max(round_delays) if round_delays else 0.0
        )
        self.per_round_total_wire_bytes.append(sum(round_wire_bytes))

    def link_utilization(
            self,
            src: int, dst: int,
            round_idx: int,
            link: LinkBandwidth,
            round_duration: float,
    ) -> float:
        """Fraction of link capacity used during a round."""
        capacity_bytes = link.capacity_mbps * 1e6 / 8 * round_duration
        if capacity_bytes <= 0:
            return 0.0
        bytes_sent = self.per_link_bytes_sent.get((src, dst), [0.0])[round_idx]
        return bytes_sent / capacity_bytes

    def bandwidth_efficiency(self, round_idx: Optional[int] = None) -> float:
        """useful_bytes / wire_bytes — fraction of bandwidth carrying useful data.

        If round_idx is None, computes across all rounds aggregated.
        """
        total_useful = 0
        total_wire = 0
        for key, useful_list in self.per_link_useful_bytes.items():
            wire_list = self.per_link_bytes_sent[key]
            if round_idx is None:
                total_useful += sum(useful_list)
                total_wire += sum(wire_list)
            elif round_idx < len(useful_list):
                total_useful += useful_list[round_idx]
                total_wire += wire_list[round_idx]
        if total_wire == 0:
            return 0.0
        return total_useful / total_wire

    def summary(self) -> dict:
        all_caps = []
        all_utils = []
        for key, bytes_list in self.per_link_bytes_sent.items():
            all_caps.extend(bytes_list)
        return {
            "mean_bytes_per_link_per_round": float(np.mean(all_caps)) if all_caps else 0.0,
            "aggregate_wire_bytes": int(sum(self.per_round_total_wire_bytes)),
            "mean_round_wall_clock_sec": float(np.mean(self.per_round_max_delay))
            if self.per_round_max_delay else 0.0,
            "overall_efficiency": self.bandwidth_efficiency(),
        }