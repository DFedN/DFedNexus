"""
simulation.py
Asynchronous DFL simulation engines.

Broadcast now passes a coordinate-level arrival mask through the inbox.
Aggregators that need it (PCG) read it from inbox tuple position 3.
Aggregators that do not need it (gossip, push-sum) ignore positions 2 and 3.
"""
from __future__ import annotations

import copy
import heapq
import logging
import time
import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cvxpy as cp
import numpy as np
import torch
from torch.utils.data import DataLoader

from .aggregators import make_aggregator, is_push_sum, is_zerofill, is_eden, is_tcg
from .communication import make_recv_fn, compute_packet_layout
from .data import make_loader
from .metrics import (
    MetricsTracker, consensus_distance,
    mean_weight, approx_grad_norm_sq,
    mean_inbox_aoi, max_inbox_aoi, update_reject_rate,
)
from .models import get_model
from .node import Node
from .bandwidth import BandwidthMatrix, BandwidthMetrics, compute_transmission_delay

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Async clock
# ─────────────────────────────────────────────────────────────────────────────

class AsyncClock:
    def __init__(self, num_nodes: int, speed_dist: str = "uniform",
                 speed_min: float = 0.5, speed_max: float = 2.0,
                 seed: int = 42):
        rng = np.random.default_rng(seed)
        if speed_dist == "uniform":
            self.speed = rng.uniform(speed_min, speed_max, num_nodes)
        elif speed_dist == "exponential":
            self.speed = np.clip(rng.exponential(1.0, num_nodes),
                                 speed_min, speed_max)
        elif speed_dist == "heterogeneous":
            n3 = num_nodes // 3
            s  = [speed_min]*n3 + [1.0]*n3 + [speed_max]*(num_nodes - 2*n3)
            rng.shuffle(s)
            self.speed = np.array(s[:num_nodes], dtype=float)
        else:
            self.speed = np.ones(num_nodes, dtype=float)
        self._pq  = [(0.0, i) for i in range(num_nodes)]
        heapq.heapify(self._pq)
        self.time = np.zeros(num_nodes, dtype=float)

    def next_node(self) -> Tuple[int, float]:
        vt, i        = heapq.heappop(self._pq)
        self.time[i] = vt
        return i, vt

    def advance(self, i: int):
        new_t        = self.time[i] + float(self.speed[i])
        self.time[i] = new_t
        heapq.heappush(self._pq, (new_t, i))

    def staleness(self) -> int:
        return int(self.time.max() - self.time.min())


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_nodes(method, config, train_ds, parts, init_sd, device) -> List[Node]:
    nodes = []
    for i in range(config["num_nodes"]):
        model = get_model(config.get("model", "smallcnn"), config["dataset"])
        model.load_state_dict(copy.deepcopy(init_sd))
        loader = make_loader(
            train_ds, parts[i],
            batch_size = config.get("batch_size", 64),
            pin_memory = (device.type == "cuda"),
        )
        agg = make_aggregator(method)
        agg.reset(i, config["num_nodes"], config)
        nodes.append(Node(i, model, loader, agg, device, config))
    return nodes


def _broadcast(
    src_node:  Node,
    topo,
    nodes:     List[Node],
    rng:       np.random.Generator,
    K:         int,
    packet_size_bytes: int,
    bytes_per_coord: int,
    recv_fn:   Callable,
    bw_matrix: BandwidthMatrix,
    bw_metrics: BandwidthMetrics,
    rnd:       int = 0,
    vt:        float = 0.0,
) -> Tuple[List[float], float, int, float]:
    """
    Simulate one broadcast from src_node to its out-neighbours.

    For each neighbour, the channel returns:
        (reconstructed_flat, completeness, coord_mask)

    The 4-tuple stored in dst node's inbox is:
        (flat, completeness, sender_alpha, coord_mask)
    """
    tx_time_duration = 0.0
    out_deg   = topo.out_deg(src_node.i)
    alpha_src = 1.0 / (out_deg + 1) if out_deg > 0 else 1.0
    comps     = []
    delay = 0.0
    wire_bytes = 0

    agg = src_node.agg

    if is_push_sum(agg):
        # ── Push-sum branch ─────────────────────────────
        if agg.s is None:
            theta = src_node.flat()
            agg.s = theta.copy()
            agg.w = 1.0

        w_before = float(agg.w)

        agg.s = agg.s * alpha_src
        agg.w = agg.w * alpha_src

        flat_to_send = agg.s.copy()
        sender_w     = alpha_src * w_before

        for dst_id, pkt_loss in topo.nbrs(src_node.i).items():
            q_ji = 1.0 - pkt_loss
            rv, comp, mask = recv_fn(
                flat_to_send, q_ji, packet_size_bytes, bytes_per_coord,
                nodes[dst_id].flat(), "zero_fill", rng,
            )

            nodes[dst_id].receive(src_node.i, rv, comp, sender_w, mask, gen_time=vt)
            comps.append(comp)

    else:
        tx_time_start = time.perf_counter()
        src_flat = src_node.flat()

        tx_time_duration = time.perf_counter() - tx_time_start

        for dst_id, pkt_loss in topo.nbrs(src_node.i).items():
            link = bw_matrix.get_link(src_node.i, dst_id)
            payload_bytes = src_flat.nbytes
            q_ji = 1.0 - pkt_loss

            if is_zerofill(agg):
                rv, comp, mask = recv_fn(
                    src_flat, q_ji, packet_size_bytes, bytes_per_coord,
                    nodes[dst_id].flat(), "zero_fill", rng,
                )
            else:
                rv, comp, mask = recv_fn(
                    src_flat, q_ji, packet_size_bytes, bytes_per_coord,
                    nodes[dst_id].flat(), "local_fill", rng,
                )

            # Compute transmission delay
            delay = compute_transmission_delay(
                payload_bytes=payload_bytes,
                link=link,
                round_idx=rnd,
                rng=rng,
            )

            wire_bytes = int(payload_bytes * 1.05)  # 5% protocol overhead
            useful_bytes = int(payload_bytes * comp)

            # Record metrics
            bw_metrics.record(src_node.i, dst_id, useful_bytes, wire_bytes, delay)

            nodes[dst_id].receive(src_node.i, rv, comp, alpha_src, mask, gen_time=vt)

            comps.append(comp)

    return comps, delay, wire_bytes, tx_time_duration


def _evaluate_all(nodes, test_loader):
    per_node = [nd.evaluate(test_loader) for nd in nodes]
    accs     = [a * 100 for a, _ in per_node]
    losses   = [l for _, l in per_node]
    return float(np.mean(accs)), float(np.mean(losses)), accs


def _record_and_print(
    tracker, rnd, mean_acc, mean_loss, elapsed,
    nodes, round_comps, config, test_loader, device,
    mode, method, comp_at_tx, comp_at_rx, transmission_delay, extra="",
    mean_aoi=None, max_aoi=None, reject_rate=None
):
    cons_dist = (consensus_distance([nd.flat() for nd in nodes])
                 if config.get("track_consensus", True) else None)

    grad_norm = None
    if (config.get("track_grad_norm", False)
            and rnd % (config.get("eval_every", 10) * 5) == 0):
        avg_flat = np.mean([nd.flat() for nd in nodes], axis=0)
        nodes[0].set_flat(avg_flat)
        grad_norm = approx_grad_norm_sq(
            nodes[0].model, test_loader, device)

    ws = [nd.get_push_sum_weight() for nd in nodes
          if nd.get_push_sum_weight() is not None]

    tracker.record(
        rnd       = rnd,
        accuracy  = mean_acc,
        loss      = mean_loss,
        elapsed   = elapsed,
        comp_tx   = comp_at_tx,
        comp_rx   = comp_at_rx,
        transmission_duration = transmission_delay,
        cons_dist = cons_dist,
        grad_norm = grad_norm,
        mean_w    = mean_weight(ws) if ws else None,
        mean_comp = float(np.mean(round_comps)) if round_comps else None,
        mean_aoi=mean_aoi,
        max_aoi=max_aoi,
        reject_rate=reject_rate,
    )

    cons_str = f" Δ={cons_dist:.4f}" if cons_dist is not None else ""
    w_str    = f" w̄={float(np.mean(ws)):.3f}" if ws else ""
    tag      = f"[{mode.upper()} {method:14s}]"
    print(
        f"\r{tag} rnd {rnd:4f}"
        f" | acc={mean_acc:5.1f}%"
        f" | loss={mean_loss:.4f}"
        f"{cons_str}{w_str}{extra}   ",
        end="", flush=True,
    )


def _finalise(tracker: MetricsTracker, method: str,
              mode: str, t0: float,
              bw_summary: dict = None,
              model_size: float = None,
              num_pckts: float = None,
              setup_cost: float = None) -> dict:
    total_time = time.perf_counter() - t0
    result     = tracker.to_dict()
    result["total_time_s"] = total_time
    result["mode"]         = mode
    result["method"]       = method
    result["bw_summary"] = bw_summary
    result["model_size"] = model_size
    result["num_pckts"] = num_pckts
    result["setup_cost"] = setup_cost
    s = tracker.summary()
    print(f"\n  ✓ [{mode.upper()}] {method} | "
          f"final={s['final_accuracy']:.1f}% | "
          f"max={s['max_accuracy']:.1f}% | "
          f"time={total_time/60:.1f}m")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Async engine
# ─────────────────────────────────────────────────────────────────────────────

def run_async(method, config, train_ds, test_loader, parts, topo, init_sd, device):
    # ── Termination mode ──────────────────────────────────────────────────────
    # time mode  : set simulation_time in config (virtual-second budget)
    # round mode : set num_rounds (legacy; terminates on min-node-activations)
    simulation_time = config.get("simulation_time", None)
    use_time_mode = simulation_time is not None
    simulation_time = float(simulation_time) if use_time_mode else None
    num_rounds = int(config.get("num_rounds", 200))

    # Evaluation interval
    # time mode  : eval_interval_time (virtual seconds between evals)
    # round mode : eval_every (rounds between evals, legacy)
    eval_interval_time = config.get("eval_interval_time", None)
    if use_time_mode and eval_interval_time is None:
        eval_interval_time = simulation_time / 50.0  # 50 eval points by default
    eval_interval_time = float(eval_interval_time) if eval_interval_time else None
    eval_every = int(config.get("eval_every", 10))

    # Total step budget: large in time mode (vt triggers termination),
    # limited in round mode (legacy behaviour preserved exactly).
    total_steps = (
        int(1e9) if use_time_mode
        else config["num_nodes"] * num_rounds
    )

    local_steps = int(config.get("local_steps", 1))
    K = int(config.get("K", 100))
    packet_size_bytes = config["communication"]["packet_size_bytes"]
    bytes_per_coord = config["communication"]["bytes_per_coord"]
    rng = np.random.default_rng(int(config.get("seed", 42)))
    recv_fn = make_recv_fn(config.get("channel", {"model": "bernoulli"}))
    nodes = _build_nodes(method, config, train_ds, parts, init_sd, device)
    _, number_of_packets = compute_packet_layout(len(nodes[0].flat()), packet_size_bytes, bytes_per_coord)
    model_bytes_mb = nodes[0].flat().nbytes / (1024 ** 2)

    clk = AsyncClock(
        num_nodes=config["num_nodes"],
        speed_dist=config.get("speed_dist", "uniform"),
        speed_min=float(config.get("speed_min", 0.5)),
        speed_max=float(config.get("speed_max", 2.0)),
        seed=int(config.get("seed", 42)),
    )
    ewma_log = []
    tracker = MetricsTracker()
    has_int = hasattr(topo, "step")

    acts = np.zeros(config["num_nodes"], dtype=int)
    completed_round = 0
    round_comps_buf: List[float] = []

    t0_setup = time.perf_counter()
    if "softdsgd" in method:
        if method == "softdsgd-uniform":
            mixing_matrix = topo.compute_uniform_neighbor_matrix()
        else:
            mixing_matrix = topo.compute_metropolis_hastings_matrix()
    elif method == "dpsgd" or method == "fedavg":
        mixing_matrix = topo.compute_uniform_neighbor_matrix()
    elif method == "zerofill":
        mixing_matrix = topo.compute_uniform_neighbor_matrix()
    elif method == "eden":
        mixing_matrix = topo.compute_metropolis_hastings_matrix()
    else:
        mixing_matrix = None

    setup_cost = time.perf_counter() - t0_setup

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path(config.get("output_dir", "./results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # bandwidth matrix generation for a simulation
    bandwidth_config = config["bandwidth"]  # from your YAML config
    edges = topo.get_edge_list()
    bw_matrix = BandwidthMatrix(edges, bandwidth_config, rng=rng)
    bw_metrics = BandwidthMetrics()
    with open(out_dir / "bandwidth_matrix_summary.json", "w") as f:
        json.dump(bw_matrix.summary(), f, indent=2, default=float)

    t0 = time.perf_counter()
    mode_label = (f"{simulation_time:.0f}s sim-time"
                  if use_time_mode else f"{num_rounds} rounds")
    log.info(f"[ASYNC] {method} | {mode_label} | {len(nodes)} nodes")
    print(f"[INFO] [ASYNC] {method} | {mode_label} | {len(nodes)} nodes")

    last_eval_vt = -1.0  # last virtual time at which we evaluated
    current_vt = 0.0  # monotonically non-decreasing global virtual time

    for step in range(1, total_steps + 1):
        node_id, vt = clk.next_node()
        current_vt = vt  # vt from min-heap is non-decreasing

        nd = nodes[node_id]
        nd.reencode()
        nd.train_step(local_steps)
        nd.reencode()
        comps, delay, wire_bytes, tx_dur = _broadcast(
            nd, topo, nodes, rng, K,
            packet_size_bytes, bytes_per_coord,
            recv_fn, bw_matrix, bw_metrics, vt=vt,
        )
        round_comps_buf.extend(comps)
        nd.aggregate(mixing_matrix=mixing_matrix)
        clk.advance(node_id)

        # EWMA tracking — reads q_hat after each aggregate call
        track_i = config.get("ewma_track_node_i", None)
        track_j = config.get("ewma_track_sender_j", None)
        if (track_i is not None
                and node_id == int(track_i)
                and hasattr(nd.agg, "q_hat")
                and int(track_j) in nd.agg.q_hat):
            ewma_log.append({
                "vt": round(float(vt), 3),
                "q_hat": round(float(nd.agg.q_hat[int(track_j)]), 5),
                "true_q": float(config.get("ewma_true_q", -1)),
                "beta": float(nd.agg.beta),
            })

        acts[node_id] += 1

        # ── Evaluation trigger ────────────────────────────────────────────────
        if use_time_mode:
            # Time mode: evaluate every eval_interval_time virtual seconds.
            should_eval = (current_vt >= last_eval_vt + eval_interval_time)
            final_eval = (current_vt >= simulation_time)
            eval_label = current_vt  # x-axis value = virtual seconds
        else:
            # Round mode: evaluate every eval_every COMPLETED rounds (legacy).
            new_round = int(acts.min())
            if new_round > completed_round:
                if has_int:
                    for _ in range(new_round - completed_round):
                        topo.step()
                completed_round = new_round
            should_eval = (completed_round > 0 and
                           completed_round % eval_every == 0 and
                           completed_round != getattr(_record_and_print, "_last_rnd", -1))
            final_eval = (completed_round >= num_rounds)
            eval_label = completed_round  # x-axis value = round number

        if should_eval or final_eval:
            mean_acc, mean_loss, _ = _evaluate_all(nodes, test_loader)
            elapsed = time.perf_counter() - t0
            staleness = clk.staleness()
            if use_time_mode:
                remaining = simulation_time - current_vt
                eta = elapsed * remaining / max(current_vt, 1e-9)
                extra = (f" vt={current_vt:.1f}s"
                         f" stale={staleness}"
                         f" ETA={eta / 60:.1f}m")
            else:
                eta = (elapsed / max(1, completed_round)
                       * (num_rounds - completed_round))
                extra = f" stale={staleness} ETA={eta / 60:.1f}m"


            # Compute AoI metrics at this virtual time snapshot
            _c_min = float(config.get("dflaa_c_min", config.get("c_min", 0.1)))
            _m_aoi = mean_inbox_aoi(nodes, current_vt)
            _x_aoi = max_inbox_aoi(nodes, current_vt)
            _rr = update_reject_rate(nodes, _c_min)

            _record_and_print(
                tracker, eval_label, mean_acc, mean_loss, elapsed,
                nodes, round_comps_buf, config, test_loader, device,
                "async", method, 0.0, 0.0, 0.0,
                extra=extra,
                mean_aoi=_m_aoi, max_aoi=_x_aoi, reject_rate=_rr,
            )
            round_comps_buf.clear()
            last_eval_vt = current_vt

        # ── Termination ───────────────────────────────────────────────────────
        if use_time_mode:
            if current_vt >= simulation_time:
                break
        else:
            if completed_round >= num_rounds:
                break

    print()

    result = _finalise(tracker, method, "async", t0, bw_metrics.summary(), model_bytes_mb, number_of_packets, setup_cost)
    result["time_axis"] = "virtual_time_s" if use_time_mode else "rounds"
    result["ewma_log"] = ewma_log
    return result
