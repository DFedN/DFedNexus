"""
topology.py
Communication graph for DFL simulation.
 
Topology types
──────────────
random_directed     Erdős–Rényi directed  (ring fallback for strong connectivity)
random_undirected   Erdős–Rényi undirected
ring_directed       i → (i+1)%n
ring_undirected     i ↔ (i+1)%n
bidirectional_ring  i → (i+1)%n  AND  i → (i-1)%n
fully_connected     every pair connected
star                node 0 is hub, bidirectional to all leaves
grid                2D 4-connected (directed = right+down; undirected = all 4)
expander            evenly-spaced neighbours (good spectral gap)
custom              edge list file  OR  numpy adjacency matrix
 
IntermittentTopology wraps any Topology with 2-state Markov link availability.
"""
from __future__ import annotations

from typing import Dict, Optional, Set, Tuple
import numpy as np
import cvxpy as cp


# ─────────────────────────────────────────────────────────────────────────────
# Channel models  (pluggable loss distributions)
# ─────────────────────────────────────────────────────────────────────────────

class ChannelModel:
    """Base class. Subclass to implement any loss distribution."""
    def sample_loss(self, rng: np.random.Generator, q_link: float) -> float:
        """Return packet_loss for one chunk given per-link q."""
        raise NotImplementedError


class BernoulliChannel(ChannelModel):
    """
    i.i.d. Bernoulli per chunk — each chunk lost independently.
    c_ji ~ Binomial(K, q_ji) / K.
    Standard memoryless erasure channel.
    """
    def sample_loss(self, rng, q_link):
        return q_link 


class GilbertElliottChannel(ChannelModel):
    """
    Two-state Markov (bursty) channel.
    Good state: low loss q_good.
    Bad  state: high loss q_bad.
    p_gb: Good→Bad transition probability per chunk.
    p_bg: Bad→Good transition probability per chunk.
    """
    def __init__(
            self,
            q_good: float = 0.05,
            q_bad: float = 0.50,
            p_gb: float = 0.05,
            p_bg: float = 0.10
    ):
        self.q_good = q_good
        self.q_bad  = q_bad
        self.p_gb   = p_gb
        self.p_bg   = p_bg

    def sample_chunk_mask(self, rng, num_chunks):
        """Return boolean mask of received chunks via Markov chain."""
        
        pi_good = self.p_bg / (self.p_gb + self.p_bg)
        state = "good" if rng.random() < pi_good else "bad"
        ok = np.zeros(num_chunks, dtype=bool)
        for k in range(num_chunks):
            q = self.q_good if state == "good" else self.q_bad
            ok[k] = rng.random() >= q 
            if state == "good" and rng.random() < self.p_gb:
                state = "bad"
            elif state == "bad" and rng.random() < self.p_bg:
                state = "good"
        return ok

    def sample_loss(self, rng, q_link):
        return q_link


class RayleighChannel(ChannelModel):
    """
    Rayleigh fading: SNR per chunk ~ Exponential(mean=snr_mean).
    Chunk received iff SNR > threshold.
    """
    def __init__(self, snr_mean_db: float = 10.0, threshold_db: float = 3.0):
        self.snr_mean_lin = 10 ** (snr_mean_db / 10)
        self.threshold    = 10 ** (threshold_db / 10)

    def sample_chunk_mask(self, rng, num_chunks):
        snr = rng.exponential(self.snr_mean_lin, size=num_chunks)
        return snr > self.threshold

    def sample_loss(self, rng, q_link):
        return q_link


def make_channel(name: str, **kwargs) -> ChannelModel:
    if name == "bernoulli":        return BernoulliChannel()
    if name == "gilbert_elliott":  return GilbertElliottChannel(**kwargs)
    if name == "rayleigh":         return RayleighChannel(**kwargs)
    raise ValueError(f"Unknown channel model: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# Topology
# ─────────────────────────────────────────────────────────────────────────────

class Topology:
    """
    adj[src][dst] = packet_loss  (reception_prob = 1 - packet_loss)
    """

    def __init__(
        self,
        num_nodes:  int,
        directed:   bool  = True,
        topo_type:  str   = "random_directed",
        avg_degree: int   = 3,
        loss_min:   float = 0.02,
        loss_max:   float = 0.45,
        seed:       int   = 42,
        edge_file:  Optional[str]        = None,
        adj_matrix: Optional[np.ndarray] = None,
    ):
        self.n         = num_nodes
        self.directed  = directed
        self.topo_type = topo_type
        self.rng       = np.random.default_rng(seed)
        self.lmin      = loss_min
        self.lmax      = loss_max
        self.adj: Dict[int, Dict[int, float]] = {i: {} for i in range(num_nodes)}

        t = topo_type.lower()
        build_map = {
            "random_directed":   lambda: self._random(True,  avg_degree),
            "random":            lambda: self._random(True,  avg_degree),
            "random_undirected": lambda: self._random(False, avg_degree),
            "ring_directed":     lambda: self._ring(True),
            "ring_undirected":   lambda: self._ring(False),
            "ring":              lambda: self._ring(False),
            "bidirectional_ring":lambda: self._bidir_ring(),
            "fully_connected":   lambda: self._full(),
            "star":              lambda: self._star(),
            "grid":              lambda: self._grid(directed),
            "expander":          lambda: self._expander(avg_degree),
            "custom":            lambda: self._custom(edge_file, adj_matrix),
        }
        if t not in build_map:
            raise ValueError(
                f"Unknown topology: {topo_type}. "
                f"Choose from: {list(build_map.keys())}")
        build_map[t]()

    # ── Loss sampling ─────────────────────────────────────────────────────────

    def _rl(self) -> float:
        return float(self.rng.uniform(self.lmin, self.lmax))

    def _add(self, s: int, d: int, bidir: bool = False):
        if d not in self.adj[s]:
            loss = self._rl()
            self.adj[s][d] = loss
            if bidir and s not in self.adj[d]:
                self.adj[d][s] = loss

    # ── Builders ──────────────────────────────────────────────────────────────

    def _random(self, directed: bool, deg: int):
        p = deg / max(1, self.n - 1)
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.rng.random() < p:
                    self._add(i, j, bidir=not directed)
        for i in range(self.n):
            self._add(i, (i + 1) % self.n, bidir=not directed)

    def _ring(self, directed: bool):
        for i in range(self.n):
            self._add(i, (i + 1) % self.n, bidir=not directed)

    def _bidir_ring(self):
        for i in range(self.n):
            self._add(i, (i + 1) % self.n)
            self._add(i, (i - 1) % self.n)

    def _full(self):
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self._add(i, j)

    def _star(self):
        for i in range(1, self.n):
            self._add(0, i)
            self._add(i, 0)

    def _grid(self, directed: bool):
        cols = int(np.ceil(np.sqrt(self.n)))
        rows = int(np.ceil(self.n / cols))
        idx  = lambda r, c: r * cols + c
        for r in range(rows):
            for c in range(cols):
                i = idx(r, c)
                if i >= self.n:
                    continue
                if c + 1 < cols:
                    j = idx(r, c + 1)
                    if j < self.n:
                        self._add(i, j, bidir=not directed)
                if r + 1 < rows:
                    j = idx(r + 1, c)
                    if j < self.n:
                        self._add(i, j, bidir=not directed)

    def _expander(self, deg: int):
        for i in range(self.n):
            for k in range(1, deg + 1):
                j = (i + self.n // max(1, deg) * k) % self.n
                if j != i:
                    self._add(i, j, bidir=not self.directed)
        for i in range(self.n):
            self._add(i, (i + 1) % self.n, bidir=not self.directed)

    def _custom(self, edge_file, adj_matrix):
        if edge_file:
            with open(edge_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    s, d  = int(parts[0]), int(parts[1])
                    loss  = float(parts[2]) if len(parts) > 2 else self._rl()
                    self.adj[s][d] = loss
                    if not self.directed:
                        self.adj[d][s] = loss
        elif adj_matrix is not None:
            for i in range(min(self.n, adj_matrix.shape[0])):
                for j in range(min(self.n, adj_matrix.shape[1])):
                    if i != j and adj_matrix[i, j] > 0:
                        loss = float(adj_matrix[i, j]) if adj_matrix[i, j] < 1 else self._rl()
                        self.adj[i][j] = loss
                        if not self.directed:
                            self.adj[j][i] = loss
        else:
            raise ValueError("custom topology requires edge_file or adj_matrix")

    # ── Query interface ───────────────────────────────────────────────────────

    def nbrs(self, src: int) -> Dict[int, float]:
        return self.adj.get(src, {})

    def get_edge_list(self) -> list:
        edges = []
        for src in range(self.n):
            for d, pl in self.adj.get(src, {}).items():
                if src != d and (src, d):
                    edges.append((src, d))
        return edges

    def out_deg(self, i: int) -> int:
        return len(self.adj.get(i, {}))

    def in_deg(self, i: int) -> int:
        return sum(1 for s in range(self.n) if i in self.adj.get(s, {}))

    def adjacency_matrix(self) -> np.ndarray:
        A = np.zeros((self.n, self.n))
        for s in range(self.n):
            for d, l in self.adj[s].items():
                A[s, d] = l
        return A

    def summary(self) -> dict:
        edges  = sum(len(v) for v in self.adj.values())
        losses = [l for row in self.adj.values() for l in row.values()]
        od     = [self.out_deg(i) for i in range(self.n)]
        ind_   = [self.in_deg(i)  for i in range(self.n)]
        return {
            "type":           self.topo_type,
            "directed":       self.directed,
            "num_nodes":      self.n,
            "num_edges":      edges,
            "avg_loss":       float(np.mean(losses)) if losses else 0.0,
            "avg_out_degree": float(np.mean(od)),
            "min_out_degree": int(min(od)),
            "max_out_degree": int(max(od)),
            "avg_in_degree":  float(np.mean(ind_)),
        }

    def compute_uniform_neighbor_matrix(self) -> np.ndarray:
        """
        Compute uniform-over-neighbors mixing matrix.

        W[i,j] = 1/(|N_i|+1) for j ∈ N_i ∪ {i}, 0 otherwise.

        Graph-adapted version of Soft-DSGD's uniform variant (Approach 1,
        Section III.B of Ye et al. 2022). The original paper assumes broadcast
        model with W = J = (1/N)·11^T; we adapt to graph topology by giving
        uniform weight only to actual neighbors plus self.
        """
        W = np.zeros((self.n, self.n))
        for i in range(self.n):
            neighbors = list(self.nbrs(i).keys())
            w_uniform = 1.0 / (len(neighbors) + 1)
            for j in neighbors:
                W[i, j] = w_uniform
            W[i, i] = w_uniform
        return W


    def compute_metropolis_hastings_matrix(self) -> np.ndarray:
        """
        Compute the Metropolis-Hastings mixing matrix W for the topology.

        The MH weights are defined as:
            W[i, j] = 1 / (1 + max(deg(i), deg(j)))     if (i, j) is an edge, i ≠ j
            W[i, j] = 0                                  if (i, j) is not an edge, i ≠ j
            W[i, i] = 1 - Σ_{j ≠ i} W[i, j]              for the diagonal

        Properties:
            - Symmetric: W^T = W (assuming symmetric topology)
            - Doubly stochastic: rows and columns each sum to 1
            - Self-weight is non-negative since 1/(1+max(deg)) ≤ 1/(1+deg(i))

        Reference: Boyd, Ghosh, Prabhakar, Shah (2006),
        "Randomized gossip algorithms," IEEE Trans. Info. Theory.

        Returns:
            W matrix of shape (N, N), symmetric and doubly stochastic.
        """
        W = np.zeros((self.n, self.n))

        degrees = np.array([len(self.nbrs(i)) for i in range(self.n)])

        for i in range(self.n):
            for j in self.nbrs(i).keys():
                W[i, j] = 1.0 / (1.0 + max(degrees[i], degrees[j]))

        for i in range(self.n):
            W[i, i] = 1.0 - W[i, :].sum()

        return W


# ─────────────────────────────────────────────────────────────────────────────
# Intermittent connectivity wrapper
# ─────────────────────────────────────────────────────────────────────────────

class IntermittentTopology:
    """
    Wraps Topology with per-round link availability using a 2-state Markov chain.

    Each link independently transitions:
        UP   → DOWN with probability p_down    per round
        DOWN → UP   with probability p_recover per round
    """

    def __init__(
        self,
        base_topo:   Topology,
        p_down:      float = 0.05,
        p_recover:   float = 0.50,
        seed:        int   = 99,
    ):
        self.base      = base_topo
        self.p_down    = p_down
        self.p_recover = p_recover
        self.rng       = np.random.default_rng(seed)
        self.n         = base_topo.n
        self.directed  = base_topo.directed
        self.topo_type = f"intermittent_{base_topo.topo_type}"
        self._down: Set[Tuple[int, int]] = set()

    def step(self) -> Set[Tuple[int, int]]:
        """
        Advance link states by one round.
        Returns set of currently-down links.
        """
        new_down: Set[Tuple[int, int]] = set()
        for s in range(self.n):
            for d in self.base.adj.get(s, {}):
                if (s, d) in self._down:
                    if self.rng.random() > self.p_recover:
                        new_down.add((s, d))   # stays down
                else:
                    if self.rng.random() < self.p_down:
                        new_down.add((s, d))   # goes down
        self._down = new_down
        return self._down

    def nbrs(self, src: int) -> Dict[int, float]:
        """Returns only currently-UP neighbours."""
        return {
            d: pl
            for d, pl in self.base.adj.get(src, {}).items()
            if (src, d) not in self._down
        }

    def get_edge_list(self) -> list:
        """Return all available (UP) directed edges as list of (src, dst) tuples.

        For directed topologies: returns one tuple per directed edge.
        For undirected topologies: returns both (i, j) and (j, i).

        Used by BandwidthMatrix to assign per-link bandwidth.
        """
        edges = []
        n = self.base.n
        for src in range(n):
            for d, pl in self.base.adj.get(src, {}).items():
                if src != d and (src, d) not in self._down:
                    edges.append((src, d))
        return edges

    def out_deg(self, i: int) -> int:
        return len(self.nbrs(i))

    def in_deg(self, i: int) -> int:
        return sum(1 for s in range(self.n) if i in self.nbrs(s))

    def fraction_up(self) -> float:
        total = sum(len(v) for v in self.base.adj.values())
        down  = len(self._down)
        return 1.0 - down / max(1, total)

    def summary(self) -> dict:
        s = self.base.summary()
        s["type"]        = self.topo_type
        s["p_down"]      = self.p_down
        s["p_recover"]   = self.p_recover
        s["fraction_up"] = self.fraction_up()
        return s

    def compute_uniform_neighbor_matrix(self) -> np.ndarray:
        """
        Compute uniform-over-neighbors mixing matrix.

        W[i,j] = 1/(|N_i|+1) for j ∈ N_i ∪ {i}, 0 otherwise.

        Graph-adapted version of Soft-DSGD's uniform variant (Approach 1,
        Section III.B of Ye et al. 2022). The original paper assumes broadcast
        model with W = J = (1/N)·11^T; we adapt to graph topology by giving
        uniform weight only to actual neighbors plus self.
        """
        W = np.zeros((self.n, self.n))
        for i in range(self.n):
            neighbors = list(self.nbrs(i).keys())
            w_uniform = 1.0 / (len(neighbors) + 1)
            for j in neighbors:
                W[i, j] = w_uniform
            W[i, i] = w_uniform
        return W


    def compute_metropolis_hastings_matrix(self) -> np.ndarray:
        """
        Compute the Metropolis-Hastings mixing matrix W for the topology.

        The MH weights are defined as:
            W[i, j] = 1 / (1 + max(deg(i), deg(j)))     if (i, j) is an edge, i ≠ j
            W[i, j] = 0                                  if (i, j) is not an edge, i ≠ j
            W[i, i] = 1 - Σ_{j ≠ i} W[i, j]              for the diagonal

        Properties:
            - Symmetric: W^T = W (assuming symmetric topology)
            - Doubly stochastic: rows and columns each sum to 1
            - Self-weight is non-negative since 1/(1+max(deg)) ≤ 1/(1+deg(i))

        Reference: Boyd, Ghosh, Prabhakar, Shah (2006),
        "Randomized gossip algorithms," IEEE Trans. Info. Theory.

        Returns:
            W matrix of shape (N, N), symmetric and doubly stochastic.
        """
        W = np.zeros((self.n, self.n))

        degrees = np.array([len(self.nbrs(i)) for i in range(self.n)])

        for i in range(self.n):
            for j in self.nbrs(i).keys():
                W[i, j] = 1.0 / (1.0 + max(degrees[i], degrees[j]))

        for i in range(self.n):
            W[i, i] = 1.0 - W[i, :].sum()

        return W