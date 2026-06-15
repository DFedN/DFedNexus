import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter

###############################################################################
# CONFIGURATION
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "results" / "dflaa"

LOSSY_LEVELS = [10, 20, 30, 50]   # main full-width figure (columns)
ZERO_LOSS = 0                     # compact single-column figure
DATASETS = ["emnist", "cifar10"]

NODE_COUNT = 20  # change to 40 or 80

# Fallback only (used by _add_inset when no data-derived limit is available).
TIME_LIMITS = {
    "emnist": 1000,
    "cifar10": 4000,
}

DATASET_TITLES = {
    "emnist": "EMNIST",
    "cifar10": "CIFAR-10",
}

ALPHA_VALUES = {
    "0p1": "0.1",
    "0p5": "0.5",
    "1p0": "1.0",
}

METHOD_ORDER = [
    "fedavg",
    "adpsgd",
    "swift",
    "softdsgd-uniform",
    "dflaa",
]

METHOD_LABELS = {
    "fedavg": "FedAvg",
    "adpsgd": "AD-PSGD",
    "swift": "SWIFT",
    "softdsgd-uniform": "Soft-DSGD",
    "dflaa": "DFL-AA",
}

TEXT_WIDTH = 12
COLUMN_WIDTH = TEXT_WIDTH / 2.07

###############################################################################
# Paper style.
###############################################################################

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

COLORS = {
    "fedavg":           "#7f7f7f",
    "adpsgd":           "#1f77b4",
    "swift":            "#2ca02c",
    "softdsgd-uniform": "#ff7f0e",
    "dflaa":            "#d62728",
}

LINESTYLES = {
    "fedavg":           "-",
    "adpsgd":           "-",
    "swift":            "-",
    "softdsgd-uniform": "-",
    "dflaa":            "-",
}

LINEWIDTHS = {
    "fedavg":           1.5,
    "adpsgd":           1.5,
    "swift":            1.5,
    "softdsgd-uniform": 1.5,
    "dflaa":            1.5,
}

###############################################################################
# DATA LOADING
###############################################################################

def experiment_dir(dataset, loss, nodes):
    return ROOT / f"{dataset}_{loss}_{nodes}_random_0p1_none"


def load_results(dataset, loss, nodes):
    folder = experiment_dir(dataset, loss, nodes)
    fp = folder / "results.json"
    if not fp.exists():
        print(f"[WARN] Missing file: {fp}")
        return None
    with open(fp, "r") as f:
        return json.load(f)


def load_topology(dataset, loss, nodes, topology):
    """{dataset}_{loss}_{nodes}_{topology}_0p1_none."""
    fp = ROOT / f"{dataset}_{loss}_{nodes}_{topology}_0p1_none" / "results.json"
    if not fp.exists():
        print(f"[WARN] Missing file: {fp}")
        return None
    with open(fp) as f:
        return json.load(f)


def load_tau_data(dataset, loss, nodes, tau):
    """{dataset}_{loss}_{nodes}_random_0p1_tau{tau}."""
    fp = ROOT / f"{dataset}_{loss}_{nodes}_random_0p1_tau{tau}" / "results.json"
    if not fp.exists():
        print(f"[WARN] Missing file: {fp}")
        return None
    with open(fp) as f:
        return json.load(f)


def load_beta_data(dataset, loss, nodes, beta):
    """{dataset}_{loss}_{nodes}_random_0p1_beta{val}."""
    fp = ROOT / f"{dataset}_{loss}_{nodes}_random_0p1_beta{beta}" / "results.json"
    if not fp.exists():
        print(f"[WARN] Missing file: {fp}")
        return None
    with open(fp) as f:
        return json.load(f)


def load_alpha_data(dataset, loss, nodes, alpha):
    """{dataset}_{loss}_{nodes}_random_{alpha}_none."""
    fp = ROOT / f"{dataset}_{loss}_{nodes}_random_{alpha}_none" / "results.json"
    if not fp.exists():
        print(f"[WARN] Missing file: {fp}")
        return None
    with open(fp) as f:
        return json.load(f)


def load_hetero_data(row_idx, loss, nodes):
    """row_idx 0 -> heterogeneous Bernoulli loss, 1 -> ResNet-18."""
    if row_idx == 0:
        fp = ROOT / f"cifar10_{loss}_{nodes}_random_0p1_hetchannel" / "results.json"
    else:
        fp = ROOT / f"cifar10_{loss}_{nodes}_random_0p1_resnet18" / "results.json"
    if not fp.exists():
        print(f"[WARN] Missing file: {fp}")
        return None
    with open(fp) as f:
        return json.load(f)

###############################################################################
# X-AXIS HELPERS  (data-derived limits, clean ticks)
###############################################################################

def _nice_ticks(xmax, n=5):
    """Return up to n evenly spaced 'round' ticks from 0 to xmax."""
    if xmax is None or xmax <= 0:
        return [0]
    raw = xmax / (n - 1)
    mag = 10 ** np.floor(np.log10(raw))
    step = mag
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * mag
        if step >= raw:
            break
    ticks = np.arange(0, xmax + step * 0.5, step)
    return ticks[ticks <= xmax * 1.001]


def _row_xmax(results, methods=METHOD_ORDER):
    """Shortest method end-time within one results dict (or None)."""
    if results is None:
        return None
    ends = [results[m]["time"][-1] for m in methods
            if m in results and results[m].get("time")]
    return min(ends) if ends else None


def _common_xmax(results_list, methods=METHOD_ORDER):
    """Min end-time across several results dicts (for shared-axis groups)."""
    xmaxes = [_row_xmax(r, methods) for r in results_list]
    xmaxes = [x for x in xmaxes if x is not None]
    return min(xmaxes) if xmaxes else None


def _set_xaxis(ax, xmax):
    """Clip x-limit to xmax with clean, non-overlapping ticks."""
    if xmax is None:
        ax.xaxis.set_major_locator(MaxNLocator(4))
        return
    ax.set_xlim(0, xmax)
    ax.set_xticks(_nice_ticks(xmax, n=5))

###############################################################################
# PLOT HELPERS
###############################################################################

def _plot_panel(ax, results, metric_key):
    plotted = False
    if results is None:
        return plotted
    for method in METHOD_ORDER:
        if method not in results:
            continue
        md = results[method]
        if metric_key not in md:
            continue
        x, y = md.get("time", []), md.get(metric_key, [])
        if len(x) == 0 or len(y) == 0 or len(x) != len(y):
            continue
        ax.plot(x, y,
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=LINEWIDTHS[method],
                label=METHOD_LABELS[method])
        plotted = True
    return plotted


def _apply_yscale(ax, yscale, linthresh=1.0, linscale=1.0):
    if yscale == "symlog":
        ax.set_yscale("symlog", linthresh=linthresh, linscale=linscale)
        ax.grid(True, which="both", linewidth=0.5, alpha=0.3)
    elif yscale == "log":
        # Consensus distance is non-negative -> plain log avoids the phantom
        # negative ticks that symlog introduces around zero.
        ax.set_yscale("log")
        ax.grid(True, which="both", linewidth=0.5, alpha=0.3)
    else:
        ax.grid(True, linewidth=0.5, alpha=0.3)


def _add_inset(ax, results, metric_key, xmax, spec):
    """Magnify a y-band in a small linear inset to separate near-zero curves."""
    axins = ax.inset_axes(spec.get("bbox", [0.35, 0.40, 0.60, 0.50]))
    _plot_panel(axins, results, metric_key)
    xlim = spec.get("xlim") or (0, xmax if xmax else 1)
    axins.set_xlim(*xlim)
    axins.set_ylim(*spec["ylim"])
    axins.set_xticklabels([])
    axins.yaxis.set_major_locator(MaxNLocator(3))
    axins.tick_params(labelsize=8)
    axins.grid(True, linewidth=0.4, alpha=0.3)
    ax.indicate_inset_zoom(axins, edgecolor="0.45", linewidth=0.8)


def _add_legend_and_save(fig, output_file, legend_ncol):
    legend_handles = [
        Line2D([0], [0],
               color=COLORS[m], linestyle=LINESTYLES[m],
               linewidth=LINEWIDTHS[m], label=METHOD_LABELS[m])
        for m in METHOD_ORDER
    ]
    n_legend_rows = -(-len(METHOD_ORDER) // legend_ncol)
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=legend_ncol, frameon=False)
    top = 1.0 - 0.055 * n_legend_rows
    fig.tight_layout(rect=[0, 0, 1, top])
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

def _apply_xaxis_format(ax, offset = 1.0):
    """Force 10^3, 10^4 style formatting with offset outside plot."""
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((2, 3))
    ax.xaxis.set_major_formatter(formatter)
    # Move offset text to the right side, outside the plot
    ax.xaxis.get_offset_text().set_x(offset)
    ax.xaxis.get_offset_text().set_horizontalalignment('right')

###############################################################################
# MAIN FIGURE  (rows = datasets, cols = loss levels; sharey per row)
###############################################################################

def create_metric_figure(metric_key, y_label, output_file,
                         loss_levels, datasets, row_axis="dataset",
                         fig_width=TEXT_WIDTH, fig_height=5.5,
                         yscale="linear", linthresh=1.0, linscale=1.0,
                         legend_ncol=5):
    if row_axis == "dataset":
        rows, cols = datasets, loss_levels
        sharex, sharey = False, "row"
    else:
        rows, cols = loss_levels, datasets
        sharex, sharey = False, False

    nrows, ncols = len(rows), len(cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height),
                             sharex=sharex, sharey=sharey, squeeze=False)

    plotted_any = False

    # **NEW**: Pre-compute x-max for each ROW to ensure consistency
    row_xmax = []
    for r, rkey in enumerate(rows):
        xmaxes = []
        for c, ckey in enumerate(cols):
            dataset, loss = ((rkey, ckey) if row_axis == "dataset"
                             else (ckey, rkey))
            res = load_results(dataset, loss, NODE_COUNT)
            xmax = _row_xmax(res)
            if xmax is not None:
                xmaxes.append(xmax)
        # Use minimum xmax across all columns in this row for consistent scale
        row_xmax.append(min(xmaxes) if xmaxes else None)

    for r, rkey in enumerate(rows):
        for c, ckey in enumerate(cols):
            dataset, loss = ((rkey, ckey) if row_axis == "dataset"
                             else (ckey, rkey))
            ax = axes[r][c]

            res = load_results(dataset, loss, NODE_COUNT)
            if _plot_panel(ax, res, metric_key):
                plotted_any = True

            # **MODIFIED**: Use the pre-computed row x-max instead of per-panel
            _set_xaxis(ax, row_xmax[r])
            _apply_xaxis_format(ax)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            _apply_yscale(ax, yscale, linthresh, linscale)

            if r == 0:
                ax.set_title(f"{ckey}% Loss" if row_axis == "dataset"
                             else DATASET_TITLES[ckey])
            if c == 0:
                ax.set_ylabel(DATASET_TITLES[rkey] if row_axis == "dataset"
                              else f"{rkey}% Loss")
            if r == nrows - 1:
                ax.set_xlabel("Virtual Time (s)")

    fig.supylabel(y_label)

    if not plotted_any:
        plt.close(fig)
        raise RuntimeError(
            f"No data plotted for '{metric_key}'. Check result folders under "
            f"{ROOT} and the template in experiment_dir()."
        )

    _add_legend_and_save(fig, output_file, legend_ncol)

###############################################################################
# MAIN TABLE
###############################################################################

def print_metric_table(dataset, metric_key, loss_levels, nodes):
    print(f"\n{'='*80}")
    print(f"{dataset.upper()} : {metric_key}")
    print(f"{'='*80}")
    header = f"{'Method':<15}"
    for loss in loss_levels:
        header += f"{str(loss)+'%':>15}"
    print(header)
    print("-" * len(header))
    for method in METHOD_ORDER:
        row = f"{METHOD_LABELS[method]:<15}"
        for loss in loss_levels:
            results = load_results(dataset, loss, nodes)
            if results is None or method not in results:
                row += f"{'-':>15}"
                continue
            if metric_key in ("accuracy", "loss", "cons_dist"):
                value = results[method].get(metric_key)[-1]
            else:
                value = results[method].get(metric_key)
            if value is None:
                row += f"{'-':>15}"
            elif isinstance(value, float):
                row += f"{value:>15.2f}"
            else:
                row += f"{str(value):>15}"
        print(row)

###############################################################################
# TOPOLOGY TABLE
###############################################################################

def print_topo_metric_table(dataset, metric_key, loss, nodes,
                             topologies=(("grid", "Fully Connected"),
                                         ("ring", "Ring"))):
    print(f"\n{'='*80}")
    print(f"{dataset.upper()} : {metric_key}")
    print(f"{'='*80}")
    header = f"{'Method':<15}"
    for suffix, label in topologies:
        header += f"{str(label):>15}"
    print(header)
    print("-" * len(header))
    for method in METHOD_ORDER:
        row = f"{METHOD_LABELS[method]:<15}"
        for suffix, label in topologies:
            results = load_topology(dataset, loss, nodes, suffix)
            if results is None or method not in results:
                row += f"{'-':>15}"
                continue
            if metric_key in ("accuracy", "loss", "cons_dist"):
                value = results[method].get(metric_key)[-1]
            else:
                value = results[method].get(metric_key)
            if value is None:
                row += f"{'-':>15}"
            elif isinstance(value, float):
                row += f"{value:>15.2f}"
            else:
                row += f"{str(value):>15}"
        print(row)

###############################################################################
# TAU TABLE
###############################################################################

def print_tau_metric_table(dataset, metric_key, loss, nodes, tau):
    print(f"\n{'='*80}")
    print(f"{dataset.upper()} : {metric_key}")
    print(f"{'='*80}")
    header = f"{'Method':<15}"
    for val in tau:
        header += f"{str(val):>15}"
    print(header)
    print("-" * len(header))
    for method in METHOD_ORDER:
        row = f"{METHOD_LABELS[method]:<15}"
        for val in tau:
            results = load_tau_data(dataset, loss, nodes, val)
            if results is None or method not in results:
                row += f"{'-':>15}"
                continue
            if metric_key in ("accuracy", "loss", "cons_dist"):
                value = results[method].get(metric_key)[-1]
            else:
                value = results[method].get(metric_key)
            if value is None:
                row += f"{'-':>15}"
            elif isinstance(value, float):
                row += f"{value:>15.2f}"
            else:
                row += f"{str(value):>15}"
        print(row)

###############################################################################
# SCALABILITY FIGURE  (rows = node counts, cols = metric x dataset; sharex col)
###############################################################################

def create_nodecount_figure(output_file, loss,
                            node_counts=(20, 40, 80),
                            datasets=("emnist", "cifar10"),
                            metrics=None,
                            fig_width=TEXT_WIDTH, fig_height=7.5,
                            legend_ncol=5):
    if metrics is None:
        metrics = [
            {"key": "accuracy",  "title": "Accuracy (%)",       "yscale": "linear"},
            {"key": "cons_dist", "title": "Consensus Distance", "yscale": "log"},
        ]

    columns = [dict(m, dataset=ds) for m in metrics for ds in datasets]

    nrows, ncols = len(node_counts), len(columns)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height),
                             sharex=False, sharey="col", squeeze=False)

    # One common x-limit per column (shared down the rows).
    col_xmax = []
    for col in columns:
        ds = col["dataset"]
        res_list = [load_results(ds, loss, n) for n in node_counts]
        col_xmax.append(_common_xmax(res_list))

    plotted_any = False

    for r, nodes in enumerate(node_counts):
        for c, col in enumerate(columns):
            dataset = col["dataset"]
            ax = axes[r][c]

            if _plot_panel(ax, load_results(dataset, loss, nodes), col["key"]):
                plotted_any = True

            res = load_results(dataset, loss, nodes)
            _set_xaxis(ax, _row_xmax(res))
            _apply_xaxis_format(ax)
            # _set_xaxis(ax, col_xmax[c])     # common limit for the whole column
            ax.yaxis.set_major_locator(MaxNLocator(4))
            _apply_yscale(ax, col.get("yscale", "linear"),
                          col.get("linthresh", 1.0), col.get("linscale", 1.0))

            if r == 0:
                ax.set_title(f"{col['title']}\n{DATASET_TITLES[dataset]}")
            if c == 0:
                ax.set_ylabel(f"{nodes} Nodes")
            if r == nrows - 1:
                ax.set_xlabel("Virtual Time (s)")

    if not plotted_any:
        plt.close(fig)
        raise RuntimeError(
            f"No data plotted at loss={loss}. Check result folders under "
            f"{ROOT} and the "
            f"'{{dataset}}_{{loss}}_{{nodes}}_random_0p1_none' naming."
        )

    _add_legend_and_save(fig, output_file, legend_ncol)

###############################################################################
# EWMA FIGURE
###############################################################################

def plot_qhat_convergence(output_file, dataset, loss, nodes, betas,
                          fig_width=COLUMN_WIDTH, fig_height=5.5):
    fig, axes = plt.subplots(3, 2, figsize=(fig_width, fig_height))
    axes = axes.flatten()

    plotted_any = False
    global_true_q = None

    # First pass: collect all q_hat values for a shared y-axis.
    all_qh = []
    all_data = {}
    for beta in betas:
        results = load_beta_data(dataset, loss, nodes, beta)
        if results is None:
            all_data[beta] = None
            continue
        results = results.get("dflaa")
        if results is None or not results.get("ewma_log"):
            all_data[beta] = None
            continue
        log = results["ewma_log"]
        all_data[beta] = log
        all_qh.extend([rec["q_hat"] for rec in log])
        if global_true_q is None:
            global_true_q = log[0].get("true_q")

    if all_qh and global_true_q is not None:
        y_min = min(min(all_qh), global_true_q)
        y_max = max(max(all_qh), global_true_q)
        pad = (y_max - y_min) * 0.1
        y_lo, y_hi = y_min - pad, y_max + pad
    else:
        y_lo, y_hi = None, None

    for idx, beta in enumerate(betas):
        ax = axes[idx]
        log = all_data.get(beta)

        if log is None:
            ax.set_title(fr"$\beta=0.{beta}$ (no data)")
            ax.axis("off")
            continue

        vt = [rec["vt"] for rec in log]
        qh = [rec["q_hat"] for rec in log]

        ax.plot(vt, qh, linewidth=1.2, color="tab:blue")

        if global_true_q is not None:
            ax.axhline(global_true_q, linestyle=":", color="0.25", linewidth=1.2)

        if y_lo is not None:
            ax.set_ylim(y_lo, y_hi)

        _apply_xaxis_format(ax, 1.0)
        ax.set_title(fr"$\beta = 0.{beta}$")
        ax.set_xlabel("Virtual Time (s)")
        ax.set_ylabel(r"$\hat{q}$")
        ax.grid(True, linewidth=0.5, alpha=0.3)
        plotted_any = True

    for j in range(len(betas), 6):
        axes[j].axis("off")

    if not plotted_any:
        plt.close(fig)
        raise RuntimeError(
            f"No ewma_log data for betas={betas} "
            f"(dataset={dataset}, loss={loss}, nodes={nodes})."
        )

    fig.tight_layout()
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

###############################################################################
# CIFAR ALPHA FIGURE  (rows = alpha, cols = metric; independent x per row)
###############################################################################

def create_alpha_variant_figure(output_file, metrics, dataset, loss, alphas,
                                fig_width=COLUMN_WIDTH, fig_height=5.0,
                                legend_ncol=3):
    nrows, ncols = len(alphas), len(metrics)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height),
                             squeeze=False)

    plotted_any = False

    for r, alpha in enumerate(alphas):
        results = load_alpha_data(dataset, loss, NODE_COUNT, alpha)
        row_xmax = _row_xmax(results)   # same x for both columns of this row

        for c, m in enumerate(metrics):
            mkey = m["key"]
            ys = m.get("yscale", "linear")
            lt = m.get("linthresh", 1.0)
            ls = m.get("linscale", 1.0)
            ax = axes[r][c]

            if _plot_panel(ax, results, mkey):
                plotted_any = True

            _set_xaxis(ax, row_xmax)
            _apply_xaxis_format(ax)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            _apply_yscale(ax, ys, lt, ls)

            if "inset" in m:
                _add_inset(ax, results, mkey, row_xmax, m["inset"])

            if r == 0:
                ax.set_title(m["title"])
            if c == 0:
                ax.set_ylabel(f"Dir(\u03B1)={ALPHA_VALUES[alpha]}")
            if r == nrows - 1:
                ax.set_xlabel("Virtual Time (s)")

    if not plotted_any:
        plt.close(fig)
        raise RuntimeError(
            f"No data plotted at loss={loss}. Check result folders under "
            f"{ROOT} and the template in experiment_dir()."
        )

    _add_legend_and_save(fig, output_file, legend_ncol)

###############################################################################
# COMBINED FIGURE  (rows = datasets, cols = metric; independent x per row)
###############################################################################

def create_combined_figure(output_file, metrics, datasets, loss,
                           fig_width=COLUMN_WIDTH, fig_height=5.0,
                           legend_ncol=3):
    nrows, ncols = len(datasets), len(metrics)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height),
                             squeeze=False)

    plotted_any = False

    for r, dataset in enumerate(datasets):
        results = load_results(dataset, loss, NODE_COUNT)
        row_xmax = _row_xmax(results)

        for c, m in enumerate(metrics):
            mkey = m["key"]
            ys = m.get("yscale", "linear")
            lt = m.get("linthresh", 1.0)
            ls = m.get("linscale", 1.0)
            ax = axes[r][c]

            if _plot_panel(ax, results, mkey):
                plotted_any = True

            _set_xaxis(ax, row_xmax)
            _apply_xaxis_format(ax)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            _apply_yscale(ax, ys, lt, ls)

            if "inset" in m:
                _add_inset(ax, results, mkey, row_xmax, m["inset"])

            if r == 0:
                ax.set_title(m["title"])
            if c == 0:
                ax.set_ylabel(f"{DATASET_TITLES[dataset]}")
            if r == nrows - 1:
                ax.set_xlabel("Virtual Time (s)")

    if not plotted_any:
        plt.close(fig)
        raise RuntimeError(
            f"No data plotted at loss={loss}. Check result folders under "
            f"{ROOT} and the template in experiment_dir()."
        )

    _add_legend_and_save(fig, output_file, legend_ncol)

###############################################################################
# COMBINED HETERO / MODEL FIGURE  (each row independent: clip per row)
###############################################################################

def create_combined_hetero_figure(output_file, metrics, row_count, loss,
                                  fig_width=COLUMN_WIDTH, fig_height=5.0,
                                  legend_ncol=3):
    nrows, ncols = row_count, len(metrics)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height),
                             squeeze=False)

    plotted_any = False

    for r in range(row_count):
        results = load_hetero_data(r, loss, NODE_COUNT)
        row_name = "Het. Bernoulli Loss" if r == 0 else "ResNet18-Architecture"
        # Each row has its own time scale -> clip to that row's shortest method.
        row_xmax = _row_xmax(results)

        for c, m in enumerate(metrics):
            mkey = m["key"]
            ys = m.get("yscale", "linear")
            lt = m.get("linthresh", 1.0)
            ls = m.get("linscale", 1.0)
            ax = axes[r][c]

            if _plot_panel(ax, results, mkey):
                plotted_any = True

            _set_xaxis(ax, row_xmax)
            _apply_xaxis_format(ax)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            _apply_yscale(ax, ys, lt, ls)

            if "inset" in m:
                _add_inset(ax, results, mkey, row_xmax, m["inset"])

            if r == 0:
                ax.set_title(m["title"])
            if c == 0:
                ax.set_ylabel(row_name)
            if r == nrows - 1:
                ax.set_xlabel("Virtual Time (s)")

    if not plotted_any:
        plt.close(fig)
        raise RuntimeError(f"No data plotted at loss={loss}.")

    _add_legend_and_save(fig, output_file, legend_ncol)

###############################################################################
# GENERATE FIGURES
###############################################################################

if __name__ == "__main__":

    out_dir = ROOT / "paper_experiments"

    # ---- Main: full-width, 2 rows (datasets) x 4 cols (losses 10-50)
    create_metric_figure(
        "accuracy", "Accuracy (%)",
        out_dir / f"hewa_kaluannakkage-fig4.pdf",
        loss_levels=LOSSY_LEVELS, datasets=DATASETS, row_axis="dataset",
        fig_width=TEXT_WIDTH, fig_height=5.5, legend_ncol=5)
    create_metric_figure(
        "cons_dist", "Consensus Distance",
        out_dir / f"hewa_kaluannakkage-fig5.pdf",
        loss_levels=LOSSY_LEVELS, datasets=DATASETS, row_axis="dataset",
        fig_width=TEXT_WIDTH, fig_height=5.5, legend_ncol=5,
        yscale="log")

    # ---- Scalability: 3 rows (node levels) x 4 cols (2 metric x 2 dataset)
    create_nodecount_figure(
        out_dir / f"hewa_kaluannakkage-fig6.pdf",
        loss=10)

    # ---- Zero loss: compact 2x2 (datasets x [accuracy, consensus])
    create_combined_figure(
        out_dir / f"zeroloss_combined_{NODE_COUNT}nodes.pdf",
        metrics=[
            {"key": "accuracy",  "title": "Accuracy (%)"},
            {"key": "cons_dist", "title": "Consensus Distance",
             "yscale": "log"},
        ],
        datasets=DATASETS, loss=ZERO_LOSS,
        fig_width=COLUMN_WIDTH, fig_height=5.5, legend_ncol=3)

    # ---- Different alphas: compact 2x2 (alpha x [accuracy, consensus])
    create_alpha_variant_figure(
        out_dir / f"hewa_kaluannakkage-fig7.pdf",
        metrics=[
            {"key": "accuracy", "title": "Accuracy (%)"},
            {"key": "cons_dist", "title": "Consensus Distance",
             "yscale": "log"},
        ],
        dataset="cifar10", loss=10, alphas=["0p5", "1p0"],
        fig_width=COLUMN_WIDTH, fig_height=5.5, legend_ncol=3)

    # ---- Het. channel + ResNet-18: 2 rows x [accuracy, consensus]
    create_combined_hetero_figure(
        out_dir / f"hewa_kaluannakkage-fig8.pdf",
        metrics=[
            {"key": "accuracy", "title": "Accuracy (%)"},
            {"key": "cons_dist", "title": "Consensus Distance",
             "yscale": "log"},
        ],
        row_count=2, loss=10,
        fig_width=COLUMN_WIDTH, fig_height=5.5, legend_ncol=3)

    # ---- EWMA convergence ablation
    plot_qhat_convergence(
        out_dir / f"hewa_kaluannakkage-fig9.pdf",
        "emnist", 20, NODE_COUNT,
        betas=["01", "02", "05", "10", "20", "30"])


    # ---- Metric tables
    all_loss_level = [0] + LOSSY_LEVELS
    for ds in ("emnist", "cifar10"):
        print_metric_table(ds, "accuracy", all_loss_level, NODE_COUNT)
        print_metric_table(ds, "loss", all_loss_level, NODE_COUNT)
        print_metric_table(ds, "auc", all_loss_level, NODE_COUNT)

    # ---- Topology robustness tables
    for ds in ("emnist", "cifar10"):
        print_topo_metric_table(ds, "accuracy", 10, NODE_COUNT)
        print_topo_metric_table(ds, "loss", 10, NODE_COUNT)
        print_topo_metric_table(ds, "auc", 10, NODE_COUNT)

    # ---- Tau ablation tables
    tau_val = ["01", "02", "03", "10", "15", "20"]
    print_tau_metric_table("emnist", "accuracy", 10, NODE_COUNT, tau_val)
    print_tau_metric_table("emnist", "loss", 10, NODE_COUNT, tau_val)
    print_tau_metric_table("emnist", "auc", 10, NODE_COUNT, tau_val)
    print_tau_metric_table("emnist", "cons_dist", 10, NODE_COUNT, tau_val)