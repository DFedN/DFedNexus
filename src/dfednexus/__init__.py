"""
DFedNexus
─────────
Reliable Decentralised Federated Learning for Wireless Edge Networks.
"""
__version__ = "1.0.0"

from .models      import get_model, SmallCNN, MLP
from .data        import load_dataset, dirichlet_partition, make_loader, partition_summary
from .topology    import Topology, IntermittentTopology
from .aggregators import (
    SoftDSGD, Swift,AdPSGD,IPWGossip,PSComp, PSNaive,
    make_aggregator, is_push_sum, list_methods,
)
from .communication       import (
    recv_bernoulli, recv_gilbert_elliott, recv_rayleigh,
    make_recv_fn, to_flat, from_flat,
)
from .node        import Node
from .metrics     import (
    MetricsTracker, consensus_distance, mean_weight,
    approx_grad_norm_sq, rounds_to_target, compute_auc,
)
from .simulation  import run_async
from .plotting    import plot_all
from .bandwidth import (
    LinkBandwidth, BandwidthMatrix, compute_transmission_delay,
    mm1k_loss_probability, apply_queue_loss, BandwidthMetrics
)

__all__ = [
    # models
    "get_model", "SmallCNN", "MLP",
    # data
    "load_dataset", "dirichlet_partition", "make_loader", "partition_summary",
    # topology
    "Topology", "IntermittentTopology",
    # aggregators
    "SoftDSGD", "Swift", "AdPSGD", "IPWGossip", "PSComp", "PSNaive",
    "make_aggregator", "is_push_sum", "list_methods",
    # communication
    "recv_bernoulli", "recv_gilbert_elliott", "recv_rayleigh",
    "make_recv_fn", "to_flat", "from_flat",
    # node
    "Node",
    # metrics
    "MetricsTracker", "consensus_distance", "mean_weight",
    "approx_grad_norm_sq", "rounds_to_target", "compute_auc",
    # simulation
    "run_async",
    # plotting
    "plot_all",
    # bandwidth
    "LinkBandwidth", "BandwidthMatrix", "compute_transmission_delay",
    "mm1k_loss_probability", "apply_queue_loss", "BandwidthMetrics",
]