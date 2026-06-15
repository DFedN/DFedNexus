"""
plotting.py
Generate all paper figures from experiment results.
"""
from __future__ import annotations
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _save(fig, path: str):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")


def plot_accuracy_vs_rounds(results: dict, save_path: str, title: str = ""):
    """
    Adaptive x-axis: uses virtual time (seconds) when any result has
    time_axis == 'virtual_time_s', otherwise falls back to round numbers.
    """
    if not HAS_MPL: return

    # Determine x-axis label from results
    use_time = any(
        r.get("time_axis") == "virtual_time_s"
        for r in results.values()
    )
    xlabel = "Virtual Time (s)" if use_time else "Communication Round"

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (method, r) in enumerate(results.items()):
        xs = r.get("rounds", [])
        ys = r.get("accuracy", [])
        if not xs or len(xs) != len(ys): continue
        label = f"{method} ({r.get('mode', '?')}) — {r.get('final_accuracy', 0):.1f}%"
        ax.plot(xs, ys, color=COLORS[idx % len(COLORS)],
                marker=MARKERS[idx % len(MARKERS)], ms=3,
                lw=2, label=label)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    default_title = (
        "Test Accuracy vs Virtual Time"
        if use_time else "Test Accuracy vs Communication Rounds"
    )
    ax.set_title(title or default_title, fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    _save(fig, save_path)


def plot_consensus_distance(results: dict, save_path: str, title: str = ""):
    """
    Plot Δ(t) on log-scale — directly validates Theorem 2.
    PS-Comp should contract geometrically.
    Biased methods should converge to a non-zero floor.
    """
    if not HAS_MPL: return
    has_data = any(r.get("cons_dist") for r in results.values())
    if not has_data: return

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (method, r) in enumerate(results.items()):
        xs = r.get("rounds", [])
        ys = r.get("cons_dist", [])
        if not xs or not ys or len(xs) != len(ys): continue
        ax.semilogy(xs, ys, color=COLORS[idx % len(COLORS)],
                    marker=MARKERS[idx % len(MARKERS)], ms=3,
                    lw=2, label=method)
    use_time = any(r.get("time_axis") == "virtual_time_s" for r in results.values())
    ax.set_xlabel("Virtual Time (s)" if use_time else "Communication Round", fontsize=12)
    ax.set_ylabel("Consensus Distance Δ(t)  [log scale]", fontsize=12)
    ax.set_title(title or "Consensus Distance — Theorem 2 Validation", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    _save(fig, save_path)


def plot_push_sum_weight(results: dict, save_path: str, title: str = ""):
    """
    Plot w̄(t) — validates Proposition 2.
    PS-Comp: w̄ ≈ 1.0  (stable)
    PS-naive: w̄ → 0   (weight drain → model collapse)
    """
    if not HAS_MPL: return
    has_data = any(r.get("mean_w") for r in results.values())
    if not has_data: return

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (method, r) in enumerate(results.items()):
        xs = r.get("rounds", [])
        ys = r.get("mean_w", [])
        if not xs or not ys or len(xs) != len(ys): continue
        ax.plot(xs, ys, color=COLORS[idx % len(COLORS)],
                marker=MARKERS[idx % len(MARKERS)], ms=3,
                lw=2, label=method)
    ax.axhline(1.0, color="black", linestyle="--", lw=1, alpha=0.5, label="ideal w=1")
    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Mean Push-Sum Weight w̄", fontsize=12)
    ax.set_title(title or "Push-Sum Weight — Proposition 2 Validation", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _save(fig, save_path)


def plot_completeness(results: dict, save_path: str, title: str = ""):
    """Plot mean completeness c̄ per round — channel condition monitoring."""
    if not HAS_MPL: return
    has_data = any(r.get("mean_comp") for r in results.values())
    if not has_data: return

    fig, ax = plt.subplots(figsize=(10, 4))
    for idx, (method, r) in enumerate(results.items()):
        xs = r.get("rounds", [])
        ys = r.get("mean_comp", [])
        if not xs or not ys or len(xs) != len(ys): continue
        ax.plot(xs, ys, color=COLORS[idx % len(COLORS)], lw=1.5, alpha=0.8,
                label=method)
    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("Mean Completeness c̄", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(title or "Mean Completeness per Round", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, save_path)


def plot_summary_bar(results: dict, save_path: str, title: str = ""):
    """Bar chart of final accuracy — for the main results table figure."""
    if not HAS_MPL: return
    methods = list(results.keys())
    finals = [r.get("final_accuracy", 0) for r in results.values()]

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.2), 5))
    bars = ax.bar(methods, finals, color=COLORS[:len(methods)], edgecolor="white", lw=0.5)

    for bar, val in zip(bars, finals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Final Test Accuracy (%)", fontsize=12)
    ax.set_title(title or "Final Accuracy by Method", fontsize=13)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(finals) * 1.15)
    plt.tight_layout()
    _save(fig, save_path)


def print_summary_table(results: dict):
    """Print a formatted summary table to stdout."""
    print(f"\n{'─' * 75}")
    print(f"{'Method':<22} {'Mode':<6} {'Final%':>7} {'Max%':>7} "
          f"{'AUC':>10} {'R→50%':>7} {'Time(m)':>8}")
    print(f"{'─' * 75}")
    use_time = any(r.get("time_axis") == "virtual_time_s" for r in results.values())
    for m, r in results.items():
        if use_time:
            r50 = r.get("time_to_50")  # virtual seconds to 50% accuracy
            r50_str = f"{r50:.1f}s" if r50 else "N/A"
        else:
            r50 = r.get("rounds_to_50")
            r50_str = str(r50) if r50 else "N/A"
        print(f"{m:<22} {r.get('mode', '?'):<6}"
              f" {r.get('final_accuracy', 0):>6.1f}%"
              f" {r.get('max_accuracy', 0):>6.1f}%"
              f" {r.get('auc', 0):>10.0f}"
              f" {r50_str:>9}"
              f" {r.get('total_time_s', 0) / 60:>7.1f}m")
    print(f"{'─' * 75}")


def plot_all(results: dict, out_dir: Path):
    """Generate all standard paper figures."""
    if not HAS_MPL:
        print("matplotlib not available — skipping plots")
        return
    if not results:
        return

    plot_accuracy_vs_rounds(
        results, str(out_dir / "accuracy_vs_rounds.png"))
    plot_consensus_distance(
        results, str(out_dir / "consensus_distance.png"))
    plot_push_sum_weight(
        results, str(out_dir / "push_sum_weight.png"))
    plot_completeness(
        results, str(out_dir / "mean_completeness.png"))
    plot_summary_bar(
        results, str(out_dir / "final_accuracy_bar.png"))
    print_summary_table(results)