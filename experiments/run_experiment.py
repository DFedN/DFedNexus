"""
run_experiment.py
Main experiment driver for DFedNexus.

Usage:
    python experiments/run_experiment.py --config configs/<.yaml file name>
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from dfednexus.data       import load_dataset, dirichlet_partition, partition_summary
from dfednexus.models     import get_model
from dfednexus.topology   import Topology, IntermittentTopology
from dfednexus.simulation import run_sync, run_async
from dfednexus.plotting   import plot_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_experiment")


def run(config: dict) -> dict:
    # ── Reproducibility ───────────────────────────────────────────────────────
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu")
    print(f"\nDevice  : {device}")

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path(config.get("output_dir", "./results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"Dataset : {config['dataset']}  alpha={config['alpha']}")
    train_ds, test_ds = load_dataset(
        config["dataset"], root=config.get("data_root", "/tmp/data"))

    # common test dataset for evaluation model performances
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False, num_workers=0)
    parts = dirichlet_partition(
        train_ds, config["num_nodes"], config["alpha"], seed=seed)

    # Print partition summary
    for p in partition_summary(train_ds, parts):
        print(f"  Node {p['node_id']:2d}: {p['num_samples']:5d} samples  "
              f"dominant class {p['dominant_class']} ({p['dominant_pct']}%)")

    # ── Topology ──────────────────────────────────────────────────────────────
    base_topo = Topology(
        num_nodes  = config["num_nodes"],
        directed   = config.get("directed", True),
        topo_type  = config.get("topology", "random_directed"),
        avg_degree = config.get("avg_degree", 3),
        loss_min   = config.get("loss_min", 0.02),
        loss_max   = config.get("loss_max", 0.45),
        seed       = seed,
        edge_file  = config.get("edge_file", None),
    )

    if config.get("intermittent", False):
        topo = IntermittentTopology(
            base_topo,
            p_down    = float(config.get("p_down",    0.05)),
            p_recover = float(config.get("p_recover", 0.50)),
            seed      = seed + 1,
        )
    else:
        topo = base_topo

    topo.summary()

    # ── Shared initial weights ────────────────────────────────────────────────
    ref_model = get_model(config.get("model", "smallcnn"), config["dataset"])
    init_sd   = copy.deepcopy(ref_model.state_dict())

    d = sum(p.numel() for p in ref_model.parameters())
    print(f"Model   : {config.get('model','smallcnn')}  ({d:,} params)")

    # ── Save config ───────────────────────────────────────────────────────────
    with open(out_dir / "config.json", "w") as f:
        json.dump({k: v for k, v in config.items() if k != "adj_matrix"},
                  f, indent=2)

    # ── Run each method ───────────────────────────────────────────────────────
    mode    = config.get("mode", "sync").lower()
    engine  = run_async if mode == "async" else run_sync
    methods = config.get("methods", ["push_sum", "softDSGD"])
    results = {}

    for method in methods:
        print(f"\n{'─'*55}\n  {method}  [{mode}]\n{'─'*55}")
        try:
            results[method] = engine(
                method      = method,
                config      = config,
                train_ds    = train_ds,
                test_loader = test_loader,
                parts       = parts,
                topo        = topo,
                init_sd     = init_sd,
                device      = device,
            )
        except Exception as e:
            log.error(f"Method {method} failed: {e}", exc_info=True)

    # ── Save results ──────────────────────────────────────────────────────────
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"{'Method':<22} {'Mode':<6} {'Final%':>7} {'Max%':>7} {'Time(m)':>8}")
    print(f"{'─'*55}")
    for m, r in results.items():
        print(f"{m:<22} {r.get('mode','?'):<6}"
              f" {r.get('final_accuracy',0):>6.1f}%"
              f" {r.get('max_accuracy',0):>6.1f}%"
              f" {r.get('total_time_s',0)/60:>7.1f}m")
    print(f"{'='*55}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_all(results, out_dir)

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--nodes",      type=int)
    p.add_argument("--rounds",     type=int)
    p.add_argument("--alpha",      type=float)
    p.add_argument("--topology",   type=str)
    p.add_argument("--directed",   dest="directed", action="store_true",  default=None)
    p.add_argument("--undirected", dest="directed", action="store_false")
    p.add_argument("--mode",       choices=["sync","async"])
    p.add_argument("--methods",    nargs="+")
    p.add_argument("--output",     type=str)
    p.add_argument("--seed",       type=int)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    overrides = {
        "num_nodes":  args.nodes,
        "num_rounds": args.rounds,
        "alpha":      args.alpha,
        "topology":   args.topology,
        "directed":   args.directed,
        "mode":       args.mode,
        "methods":    args.methods,
        "output_dir": args.output,
        "seed":       args.seed,
    }
    for k, v in overrides.items():
        if v is not None:
            config[k] = v

    run(config)


if __name__ == "__main__":
    main()