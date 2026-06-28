"""
Aggregate the input-ablation results and build the comparison tables, paired
significance tests, and dose-response plots.

Baseline (B) = the ETH FOV model: tag `eth_vel_fov_gpu_12`.
Ablation tags (all FOV=200, 100 epochs, ETH):
    E_r2     -> eth_fov_abl_r2_12        (radius 2.0,  H=7)
    E_none   -> eth_fov_abl_noedge_12    (no edges,    H=7)
    H3_tr    -> eth_fov_abl_H3train_12   (H=3, retrained)
    H1_tr    -> eth_fov_abl_H1train_12   (H=1, retrained)
    H5_inf   -> eth_fov_abl_H5infer_12   (H=5, inference-only)
    H3_inf   -> eth_fov_abl_H3infer_12   (H=3, inference-only)
    H1_inf   -> eth_fov_abl_H1infer_12   (H=1, inference-only)

Each evaluate.py CSV has one row per prediction instance in a `value` column.
most_likely/best_of have 364 rows (naturally paired across runs -> Wilcoxon);
z_mode/full have 2000x more rows (means only).

Usage (from anywhere, paths are resolved relative to this file):
    python experiments/ablation/aggregate_ablation.py
    python experiments/ablation/aggregate_ablation.py --results_dir /path/to/results
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.normpath(os.path.join(HERE, "..", "pedestrians", "results"))

BASELINE = ("B (FOV, r=3.0, H=7)", "eth_vel_fov_gpu_12")
RUNS = [
    ("E_r2 (r=2.0)",      "eth_fov_abl_r2_12"),
    ("E_none (no edges)", "eth_fov_abl_noedge_12"),
    ("H3_tr (H=3 train)", "eth_fov_abl_H3train_12"),
    ("H1_tr (H=1 train)", "eth_fov_abl_H1train_12"),
    ("H5_inf (H=5 infer)", "eth_fov_abl_H5infer_12"),
    ("H3_inf (H=3 infer)", "eth_fov_abl_H3infer_12"),
    ("H1_inf (H=1 infer)", "eth_fov_abl_H1infer_12"),
]
MODES = ["most_likely", "z_mode", "best_of", "full"]
METRICS = ["ade", "fde", "kde"]


def load_values(results_dir, tag, metric, mode):
    path = os.path.join(results_dir, f"{tag}_{metric}_{mode}.csv")
    if not os.path.exists(path):
        return None
    v = pd.read_csv(path)["value"].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return v if len(v) else None


def mean_or_nan(results_dir, tag, metric, mode):
    v = load_values(results_dir, tag, metric, mode)
    return float(np.mean(v)) if v is not None else float("nan")


def build_summary(results_dir):
    rows = []
    all_tags = [BASELINE] + RUNS
    for label, tag in all_tags:
        row = {"run": label}
        for metric in METRICS:
            for mode in MODES:
                if metric == "kde" and mode == "most_likely":
                    continue
                row[f"{metric}_{mode}"] = mean_or_nan(results_dir, tag, metric, mode)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tests(results_dir):
    """Paired Wilcoxon vs baseline on the 364-row modes (most_likely, best_of)."""
    out = []
    for metric in ("ade", "fde"):
        for mode in ("most_likely", "best_of"):
            base = load_values(results_dir, BASELINE[1], metric, mode)
            if base is None:
                continue
            for label, tag in RUNS:
                vals = load_values(results_dir, tag, metric, mode)
                if vals is None or len(vals) != len(base):
                    continue
                diff = vals - base
                d_mean = float(np.mean(diff))
                try:
                    stat, p = wilcoxon(vals, base)
                    p = float(p)
                except ValueError:
                    p = float("nan")  # all-zero differences
                sig = "n/a" if np.isnan(p) else ("**sig**" if p < 0.05 else "ns")
                out.append({
                    "metric": metric.upper(), "mode": mode, "run": label,
                    "baseline_mean": float(np.mean(base)), "run_mean": float(np.mean(vals)),
                    "delta": d_mean, "wilcoxon_p": p, "verdict": sig,
                })
    return pd.DataFrame(out)


def plot_edge_doseresponse(results_dir, out_dir):
    # x: effective radius (no-edge plotted at 0.0), y: ADE for two modes
    pts = [("B", 3.0, BASELINE[1]), ("E_r2", 2.0, "eth_fov_abl_r2_12"),
           ("E_none", 0.0, "eth_fov_abl_noedge_12")]
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for mode in ("best_of", "most_likely"):
        xs, ys, labels = [], [], []
        for name, x, tag in pts:
            m = mean_or_nan(results_dir, tag, "ade", mode)
            if not np.isnan(m):
                xs.append(x); ys.append(m); labels.append(name)
        if xs:
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], "o-", label=f"ADE ({mode})")
            for x, y, nm in zip(xs, ys, labels):
                ax.annotate(nm, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
            plotted = True
    if not plotted:
        plt.close(fig); return None
    ax.set_xlabel("PED-PED attention radius (m); 0 = no edges")
    ax.set_ylabel("ADE (m)")
    ax.set_title("Edge dose-response (FOV held on)")
    ax.legend()
    out = os.path.join(out_dir, "ablation_edge_doseresponse.png")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def plot_history_doseresponse(results_dir, out_dir):
    base_ade = {m: mean_or_nan(results_dir, BASELINE[1], "ade", m) for m in ("best_of", "most_likely")}
    retr = {7: BASELINE[1], 3: "eth_fov_abl_H3train_12", 1: "eth_fov_abl_H1train_12"}
    infe = {7: BASELINE[1], 5: "eth_fov_abl_H5infer_12", 3: "eth_fov_abl_H3infer_12", 1: "eth_fov_abl_H1infer_12"}
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for mode in ("best_of",):  # best-of-20 is the headline metric
        for name, mapping, style in (("retrained", retr, "o-"), ("inference", infe, "s--")):
            xs = sorted(mapping)
            ys = [mean_or_nan(results_dir, mapping[h], "ade", mode) for h in xs]
            keep = [(x, y) for x, y in zip(xs, ys) if not np.isnan(y)]
            if keep:
                xx, yy = zip(*keep)
                ax.plot(xx, yy, style, label=f"{name} (ADE {mode})")
                plotted = True
    if not plotted:
        plt.close(fig); return None
    ax.set_xlabel("history length H (frames encoded)")
    ax.set_ylabel("ADE (m), best-of-20")
    ax.set_title("History dose-response (FOV held on): retrained vs inference-only")
    ax.invert_xaxis()  # fewer frames toward the right
    ax.legend()
    out = os.path.join(out_dir, "ablation_history_doseresponse.png")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=DEFAULT_RESULTS)
    ap.add_argument("--output_dir", default=None, help="default: <results_dir>/plots")
    args = ap.parse_args()
    out_dir = args.output_dir or os.path.join(args.results_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"results_dir = {args.results_dir}\n")

    summary = build_summary(args.results_dir)
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print("=== Mean metrics by run (NaN = CSV not present yet) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    summary.to_csv(os.path.join(out_dir, "ablation_summary.csv"), index=False)

    print("\n=== Paired Wilcoxon vs baseline (most_likely & best_of; n=364) ===")
    tests = paired_tests(args.results_dir)
    if len(tests):
        print(tests.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        tests.to_csv(os.path.join(out_dir, "ablation_paired_tests.csv"), index=False)
    else:
        print("(no comparable CSVs found yet)")

    e = plot_edge_doseresponse(args.results_dir, out_dir)
    h = plot_history_doseresponse(args.results_dir, out_dir)
    for p in (e, h):
        if p:
            print("saved", p)
    print(f"\nWrote summary + tests + plots to {out_dir}")


if __name__ == "__main__":
    main()
