"""
Plot the DISTRIBUTION of per-instance prediction errors produced by evaluate.py.

Each evaluate.py output CSV (e.g. eth_vel_fov_gpu_12_ade_best_of.csv) has one row
per prediction instance in a `value` column -- that column *is* the empirical error
distribution. This script turns those into density plots:

    x-axis = error value (ADE [m] / FDE [m] / KDE NLL)
    y-axis = probability density

For each metric (ADE, FDE, KDE) it overlays the available evaluation modes
(most_likely, z_mode, best_of, full) on one axes, so you can see how the error
distribution shifts between modes.

Usage (run from experiments/pedestrians/, with the conda env active):

    python plot_error_distributions.py --tag eth_vel_fov_gpu_12

    # all 5 datasets in one go:
    for ds in eth hotel univ zara1 zara2; do \
        python plot_error_distributions.py --tag ${ds}_vel_fov_gpu_12; done

Outputs PNGs to results/plots/ (headless-safe; works over SSH).
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend (no display needed on the cluster)
import matplotlib.pyplot as plt

MODES = ["most_likely", "z_mode", "best_of", "full"]
METRICS = {"ade": "ADE (m)", "fde": "FDE (m)", "kde": "KDE NLL"}


def load_values(results_dir, tag, metric, mode):
    path = os.path.join(results_dir, f"{tag}_{metric}_{mode}.csv")
    if not os.path.exists(path):
        return None
    vals = pd.read_csv(path)["value"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return vals if len(vals) else None


def plot_metric(ax, results_dir, tag, metric, clip_pct, bins):
    """Overlay the per-mode density for one metric. Returns True if anything plotted."""
    plotted = False
    for mode in MODES:
        vals = load_values(results_dir, tag, metric, mode)
        if vals is None:
            continue
        # Clip the long upper tail purely for x-axis readability; stats use full data.
        hi = np.percentile(vals, clip_pct)
        shown = vals[vals <= hi]
        ax.hist(
            shown, bins=bins, density=True, histtype="stepfilled", alpha=0.35,
            label=f"{mode}  (mean={vals.mean():.3f}, median={np.median(vals):.3f}, n={len(vals)})",
        )
        ax.axvline(vals.mean(), linestyle="--", linewidth=1, alpha=0.6)
        plotted = True
    if plotted:
        ax.set_xlabel(METRICS[metric])
        ax.set_ylabel("probability density")
        ax.legend(fontsize=8)
    return plotted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results", help="dir with evaluate.py CSVs")
    ap.add_argument("--tag", required=True, help="output_tag prefix, e.g. eth_vel_fov_gpu_12")
    ap.add_argument("--output_dir", default="results/plots")
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--clip_percentile", type=float, default=99.0,
                    help="trim x-axis at this percentile so long tails don't squash the plot")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # (a) one combined figure: ADE | FDE | KDE side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    any_combined = False
    for ax, metric in zip(axes, METRICS):
        ok = plot_metric(ax, args.results_dir, args.tag, metric, args.clip_percentile, args.bins)
        ax.set_title(f"{metric.upper()} distribution")
        any_combined = any_combined or ok
    if any_combined:
        fig.suptitle(f"{args.tag}: per-instance error distributions by evaluation mode", y=1.02)
        fig.tight_layout()
        combined = os.path.join(args.output_dir, f"{args.tag}_error_distributions.png")
        fig.savefig(combined, dpi=130, bbox_inches="tight")
        print("saved", combined)
    plt.close(fig)

    # (b) one standalone figure per metric (easier to read individually)
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8, 5))
        if plot_metric(ax, args.results_dir, args.tag, metric, args.clip_percentile, args.bins):
            ax.set_title(f"{args.tag}: {metric.upper()} distribution by evaluation mode")
            fig.tight_layout()
            out = os.path.join(args.output_dir, f"{args.tag}_{metric}_distribution.png")
            fig.savefig(out, dpi=130, bbox_inches="tight")
            print("saved", out)
        plt.close(fig)


if __name__ == "__main__":
    main()
