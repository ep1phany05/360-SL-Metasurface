import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----------------------------
# Metric definitions
# ----------------------------
DEPTH_METRICS: List[Tuple[str, str, str]] = [
    ("absrel", "AbsRel", "lower"),
    # ("sqrel", "SqRel", "lower"),
    # ("rmse", "RMSE", "lower"),
    ("rmse_log", "RMSE_log", "lower"),
    ("log10", "log10", "lower"),
    ("silog", "SILog", "lower"),
    ("mae", "MAE", "lower"),
    ("grad_l1", "GradL1", "lower"),
    ("delta1", "d1", "higher"),
    # ("delta2", "d2", "higher"),
    # ("delta3", "d3", "higher"),
]

NORMAL_METRICS: List[Tuple[str, str, str]] = [
    ("ang_mean", "AngMean", "lower"),
    ("ang_median", "AngMed", "lower"),
    ("ang_rmse", "AngRMSE", "lower"),
    ("acc_11", "Acc11.25", "higher"),
    ("acc_22", "Acc22.5", "higher"),
    ("acc_30", "Acc30", "higher"),
    ("cos_loss", "CosLoss", "lower"),
]

OCC_METRICS: List[Tuple[str, str, str]] = [
    ("precision", "Prec", "higher"),
    ("recall", "Recall", "higher"),
    ("f1", "F1", "higher"),
    ("iou", "IoU", "higher"),
    ("acc", "Acc", "higher"),
    ("bce", "BCE", "lower"),
]


# ----------------------------
# IO
# ----------------------------
def load_summary(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_metric(summary: Dict, group_key: str, metric_key: str) -> float:
    grp = summary.get(group_key, {}) or {}
    v = grp.get(metric_key, float("nan"))
    try:
        return float(v)
    except Exception:
        return float("nan")


# ----------------------------
# Scoring: relative-to-best (0..1, higher is better)
# ----------------------------
def scores_relative_to_best(
    summaries: List[Dict],
    group_key: str,
    metrics: List[Tuple[str, str, str]],
    eps: float = 1e-12,
    nan_fill: float = 0.0,
) -> Tuple[List[str], np.ndarray, np.ndarray, List[str]]:
    labels = [m[1] for m in metrics]
    direction = [m[2] for m in metrics]
    keys = [m[0] for m in metrics]

    raw = np.array([[get_metric(s, group_key, k) for k in keys] for s in summaries], dtype=np.float64)
    raw[~np.isfinite(raw)] = np.nan

    S = np.zeros_like(raw, dtype=np.float64)

    for j, direc in enumerate(direction):
        col = raw[:, j]
        if not np.any(np.isfinite(col)):
            S[:, j] = nan_fill
            continue

        if direc == "higher":
            best = np.nanmax(col)
            if abs(best) < eps:
                S[:, j] = nan_fill
            else:
                S[:, j] = np.clip(col / best, 0.0, 1.0)
        else:
            best = np.nanmin(col)
            denom = np.where(np.isfinite(col) & (np.abs(col) > eps), col, np.nan)
            sc = (best + eps) / (col + eps)
            # sc = np.where(np.isfinite(sc), sc, nan_fill)
            S[:, j] = np.clip(sc, 0.0, 1.0)

    return labels, S, raw, direction


def apply_gamma(scores: np.ndarray, gamma: float) -> np.ndarray:
    if gamma is None or abs(gamma - 1.0) < 1e-12:
        return scores
    # keep 0..1, best=1 stays 1, others shrink (gamma>1 amplifies gaps)
    return np.clip(scores, 0.0, 1.0) ** gamma


# ----------------------------
# Improvement summary (method0 vs method1)
# ----------------------------
def improvement_percent(raw0: np.ndarray, raw1: np.ndarray, direction: List[str], eps: float = 1e-12) -> np.ndarray:
    """
    Returns per-metric improvement of method0 over method1 (in percent).
    + means method0 better.
    For lower-better: (raw1 - raw0)/raw1
    For higher-better: (raw0 - raw1)/raw1
    """
    out = np.zeros_like(raw0, dtype=np.float64)
    for j, d in enumerate(direction):
        a = raw0[j]
        b = raw1[j]
        if not (np.isfinite(a) and np.isfinite(b)) or abs(b) < eps:
            out[j] = np.nan
            continue
        if d == "lower":
            out[j] = (b - a) / b * 100.0
        else:
            out[j] = (a - b) / b * 100.0
    return out


# ----------------------------
# Radar plotting
# ----------------------------
def polygon_area_from_polar(r: np.ndarray, theta: np.ndarray) -> float:
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    x2 = np.r_[x, x[0]]
    y2 = np.r_[y, y[0]]
    return 0.5 * abs(np.dot(x2[:-1], y2[1:]) - np.dot(y2[:-1], x2[1:]))


def auto_rmin(scores: np.ndarray) -> float:
    """
    Make small differences visible by zooming radial range.
    scores: (M,N) in [0,1]
    """
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return 0.0
    mn = float(np.min(finite))
    # If everything is already near 1, zoom hard.
    if mn > 0.985:
        return max(0.0, mn - 0.010)
    if mn > 0.95:
        return max(0.0, mn - 0.020)
    if mn > 0.85:
        return max(0.0, mn - 0.060)
    return 0.0


def set_radial_ticks(ax, rmin: float):
    rmax = 1.0
    ax.set_ylim(rmin, rmax)
    # 5 ticks including endpoints
    ticks = np.linspace(rmin, rmax, 5)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.3f}" if rmin > 0 else f"{t:.2f}" for t in ticks], fontsize=8)


def plot_radar(ax, title: str, labels: List[str], scores: np.ndarray, method_names: List[str], raw: np.ndarray, direction: List[str]):
    M, N = scores.shape
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=9)
    ax.tick_params(axis="x", pad=10)

    rmin = auto_rmin(scores)
    set_radial_ticks(ax, rmin)

    ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.9)
    ax.set_title(title, fontsize=13, pad=16)

    colors = plt.cm.tab10(np.linspace(0, 1, max(M, 3)))

    areas = []
    for i in range(M):
        r = scores[i]
        th_c = np.r_[theta, theta[0]]
        r_c = np.r_[r, r[0]]
        ax.plot(th_c, r_c, linewidth=3.0, marker="o", markersize=4.5, color=colors[i], label=method_names[i])
        ax.fill(th_c, r_c, alpha=0.14, color=colors[i])
        areas.append(polygon_area_from_polar(r, theta))

    # Improvement summary for first two methods (if available)
    if M >= 2:
        imp = improvement_percent(raw[0], raw[1], direction)
        finite_imp = imp[np.isfinite(imp)]
        avg_imp = float(np.nanmean(finite_imp)) if finite_imp.size else float("nan")
        best3 = np.argsort(-imp)[:min(3, N)]
        lines = []
        for j in best3:
            if np.isfinite(imp[j]):
                lines.append(f"{labels[j]}: {imp[j]:+.2f}%")
        box = (
            f"Area (score):\n"
            f"{method_names[0]}: {areas[0]:.3f}\n"
            f"{method_names[1]}: {areas[1]:.3f}\n\n"
            f"{method_names[0]} vs {method_names[1]}:\n"
            f"Avg: {avg_imp:+.2f}%\n" + ("\n".join(lines) if lines else "")
        )
        ax.text(1.17, 0.02, box, transform=ax.transAxes, fontsize=8.2, va="bottom")

    return areas


def main(file_list: List[str], name_list: List[str], out_path: str, dpi: int = 300, gamma: float = 6.0):
    if len(file_list) != len(name_list):
        raise ValueError("file_list and name_list must have the same length")

    summaries = [load_summary(p) for p in file_list]

    dlab, dsc, draw, ddir = scores_relative_to_best(summaries, "avg_depth", DEPTH_METRICS)
    nlab, nsc, nraw, ndir = scores_relative_to_best(summaries, "avg_normal", NORMAL_METRICS)
    olab, osc, oraw, odir = scores_relative_to_best(summaries, "avg_occlusion", OCC_METRICS)

    dsc = apply_gamma(dsc, 1.5)
    nsc = apply_gamma(nsc, 1.5)
    osc = apply_gamma(osc, gamma)

    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    ax1 = fig.add_subplot(1, 3, 1, polar=True)
    ax2 = fig.add_subplot(1, 3, 2, polar=True)
    ax3 = fig.add_subplot(1, 3, 3, polar=True)

    plot_radar(ax1, f"Depth", dlab, dsc, name_list, draw, ddir)
    plot_radar(ax2, f"Normal", nlab, nsc, name_list, nraw, ndir)
    plot_radar(ax3, f"Occlusion", olab, osc, name_list, oraw, odir)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(name_list), 4), frameon=False, fontsize=11)

    out_path = str(Path(out_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Radar comparison", fontsize=15, y=1.02)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", default=None, help="metrics_summary.json paths")
    parser.add_argument("--names", nargs="+", default=None, help="names for legend")
    parser.add_argument("--out", type=str, default="vis/260125/radar_metrics.png")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--gamma", type=float, default=1.5, help="Amplify gaps: score := score**gamma (gamma>1).")
    args = parser.parse_args()

    # Requested defaults:
    if args.files is None or args.names is None:
        file_list = [
            "/data/wudelong/result/360-sl-metasurface/runs/260125-172954/metrics_summary.json",
            "/data/wudelong/result/360-sl-metasurface/runs/260125-172900/metrics_summary.json"
        ]
        name_list = ["NORM", "Pixelwise"]
    else:
        file_list = args.files
        name_list = args.names

    main(file_list, name_list, args.out, dpi=args.dpi, gamma=args.gamma)
