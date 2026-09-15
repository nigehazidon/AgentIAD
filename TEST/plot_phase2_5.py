"""Phase 2.5 figures (optional): threshold sensitivity + candidate area ratio.

Reads the already-written Phase 2.5 results (no recomputation, no PatchCore) and
renders two static PNGs into ``results/phase2_5/visualizations/``:

  fig1_threshold_sensitivity.png
      one panel per measure group (never a dual axis):
        (a) defective: coverage / precision / IoU   vs threshold
        (b) normal:    candidate rate / area ratio  vs threshold
        (c) normal:    average candidate count      vs threshold

  fig2_area_ratio_distribution.png
      ECDF of the per-image Top-1 candidate area ratio, defective vs normal,
      one panel per category (small multiples).

Palette: the validated 3-slot categorical order (blue / orange / aqua) on the
light chart surface; aqua sits below 3:1 contrast, so every line carries a
visible direct label (the "relief" rule). The CSVs under results/phase2_5/ are
the table view of the same numbers.

Usage::

    /user/pfy/anaconda3/envs/anomalyagent/bin/python plot_phase2_5.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RESULT_ROOT = "/data/pfy/AgentIAD/results/phase2_5"

# --- validated palette (light surface) ------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9, length=3)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)


def spread_labels(values: list[float], gap: float) -> list[float]:
    """Nudge label positions apart so stacked direct labels never collide.

    ``values`` are final-y positions in data coordinates; returns adjusted
    positions that keep their original order but sit at least ``gap`` apart.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    for prev, cur in zip(order, order[1:]):
        if placed[cur] - placed[prev] < gap:
            placed[cur] = placed[prev] + gap
    # if we ran off the top, push the whole stack down instead
    overflow = placed[order[-1]] - max(values)
    if overflow > 0:
        for i in order:
            placed[i] -= overflow
    return placed


def line_panel(ax, xs, series: list[tuple[str, list[float]]], title: str,
               ylabel: str, ylim=(0.0, 1.0)) -> None:
    """Thin lines (2px), >=8px markers, legend + direct label at the last point."""
    for i, (name, ys) in enumerate(series):
        color = SERIES[i % len(SERIES)]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
    # direct labels (relief for low-contrast hues, and identity not color-alone),
    # placed clear of each other
    span = (ylim[1] - ylim[0]) if ylim else (max(max(s[1]) for s in series) or 1.0)
    label_y = spread_labels([s[1][-1] for s in series], gap=0.075 * span)
    for i, (name, ys) in enumerate(series):
        ax.annotate(name, xy=(xs[-1], label_y[i]), xytext=(6, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=8.5, color=INK_2,
                    arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.8,
                                    shrinkA=0, shrinkB=2)
                    if abs(label_y[i] - ys[-1]) > 1e-9 else None)
    ax.set_title(title, fontsize=10.5, color=INK, pad=10, loc="left")
    ax.set_xlabel("threshold", fontsize=9, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
    ax.set_xticks(xs)
    ax.set_xlim(min(xs) - 0.03, max(xs) + 0.16)   # room for the direct labels
    if ylim:
        ax.set_ylim(*ylim)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower left")
    style_axes(ax)


def ecdf(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    v = np.sort(np.asarray([x for x in values if np.isfinite(x)], dtype=float))
    if v.size == 0:
        return v, v
    return v, np.arange(1, v.size + 1) / v.size


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2.5 figures")
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    args = parser.parse_args()

    root = Path(args.result_root)
    summary = json.loads((root / "summary.json").read_text())
    thresholds = [float(t) for t in summary["thresholds_swept"]]
    sens = summary["threshold_sensitivity"]
    cats = summary["categories"]

    out_dir = root / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- figure 1 ---
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), facecolor=SURFACE)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.15, wspace=0.42)

    d = [sens[str(t)]["defective"] for t in thresholds]
    line_panel(axes[0], thresholds,
               [("coverage", [x["top1"]["coverage"] for x in d]),
                ("precision", [x["top1"]["precision"] for x in d]),
                ("IoU", [x["top1"]["iou"] for x in d])],
               f"(a) defective images (macro over {summary['defective']['n_images']} imgs)",
               "Top-1 metric")

    n = [sens[str(t)]["normal"] for t in thresholds]
    line_panel(axes[1], thresholds,
               [("candidate rate", [x["candidate_rate"] for x in n]),
                ("Top-1 area ratio", [x["top1_area_ratio"]["mean"] for x in n]),
                ("Top-3 area ratio", [x["top3_area_ratio"]["mean"] for x in n])],
               f"(b) normal images (macro over {summary['normal']['n_images']} imgs)",
               "value")

    line_panel(axes[2], thresholds,
               [("avg candidates", [x["average_candidate_count"] for x in n])],
               "(c) normal images: proposal count", "candidates / image",
               ylim=(1.0, max(x["average_candidate_count"] for x in n) * 1.12))

    fig.suptitle("Phase 2.5 — threshold sensitivity of PatchCore candidate regions "
                 "(min_area_ratio=0.001, max_regions=3; post-processing only, no re-fit)",
                 fontsize=11.5, color=INK, x=0.055, ha="left", y=0.94)
    fig.savefig(out_dir / "fig1_threshold_sensitivity.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)

    # ---------------------------------------------------------- figure 2 ---
    fig, axes = plt.subplots(1, len(cats), figsize=(6.4 * len(cats), 4.2),
                             facecolor=SURFACE, squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.80, top=0.80, bottom=0.14, wspace=0.30)

    for ax, cat in zip(axes[0], cats):
        metrics = json.loads((root / "category_metrics" / f"{cat}.json").read_text())
        images = metrics["images"]
        for i, (split, label) in enumerate([("defective", "defective"),
                                            ("normal", "normal")]):
            vals = [r["top1_mask_area_ratio"] for r in images if r.get("split") == split]
            x, y = ecdf(vals)
            med = float(np.median(x)) if x.size else float("nan")
            ax.plot(x, y, color=SERIES[i], linewidth=2, drawstyle="steps-post",
                    label=f"{label} (n={x.size}, median={med:.2f})", zorder=3)
            if x.size:
                # median marker only — the median value is already in the legend,
                # so a second label here would just collide with the other series
                ax.plot([med], [0.5], marker="o", markersize=6, color=SERIES[i],
                        markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        ax.set_title(f"{cat} — Top-1 candidate area ratio (threshold=0.5)",
                     fontsize=10.5, color=INK, pad=10, loc="left")
        ax.set_xlabel("candidate mask area / image area", fontsize=9, color=INK_2)
        ax.set_ylabel("fraction of images (ECDF)", fontsize=9, color=INK_2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower right")
        style_axes(ax)

    fig.suptitle("Phase 2.5 — how large is the Top-1 candidate region? "
                 "(normal images produce the LARGER regions)",
                 fontsize=11.5, color=INK, x=0.09, ha="left", y=0.93)
    fig.savefig(out_dir / "fig2_area_ratio_distribution.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)

    print(f"figures written to {out_dir}")


if __name__ == "__main__":
    main()
