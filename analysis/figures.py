#!/usr/bin/env python3
"""Plot observed transfer metrics and the source-choice regret decomposition."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
import numpy as np

INK = "#142536"
MUTED = "#5E6F7C"
BLUE = "#246C96"
GOLD = "#C9973D"
PALE = "#E9EEF1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("assets"))
    args = parser.parse_args()
    data = {name: json.loads((args.results / f"{name}.json").read_text())
            for name in ("development", "followup")}
    args.output.mkdir(parents=True, exist_ok=True)
    for weight in ("normal", "bold"):
        font_manager.findfont(
            font_manager.FontProperties(family="Times New Roman", weight=weight),
            fallback_to_default=False,
        )
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11,
                         "text.color": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": INK,
                         "svg.fonttype": "path", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.8),
                            gridspec_kw={"width_ratios": [1.10, 1], "wspace": 0.70})
    fig.subplots_adjust(left=0.20, right=0.965, bottom=0.25, top=0.76)
    fig.text(0.045, 0.935, "WHEN HARD SCENES TRANSFER", size=12,
             weight="bold", color=BLUE)
    fig.text(0.045, 0.873, "Scene agreement, target utility, and the cost of reuse", size=24, weight="bold")
    fig.text(0.045, 0.821,
             "Three camera-LiDAR detectors  /  nuScenes  /  2,007 development + 725 follow-up keyframes",
             size=12, color=MUTED)
    cells = data["development"]["outcomes"]["lost_clean_tp_count"]["fractions"]["0.20"]["cells"]
    measures = [("raw_overlap", "Scene identity\nagreement"),
                ("absolute_target_risk_coverage", "Total target-loss\ncoverage"),
                ("raw_RiskCoverage", "Target-oracle\nefficiency"),
                ("log_opportunity_adjusted_RiskCoverage", "Opportunity-adjusted\nefficiency")]
    ax = axes[0]
    ax.set_title("A   Four views of the same source-selected scenes", loc="left", pad=22, fontsize=12, weight="bold")
    for index, (key, label) in enumerate(measures):
        values = np.array([cell[key] for cell in cells], dtype=float)
        rng = np.random.default_rng(100 + index)
        jitter = rng.uniform(-0.11, 0.11, len(values))
        ax.scatter(values, index + jitter, s=15, color=BLUE, alpha=0.24, edgecolors="none")
        median = float(np.median(values))
        ax.scatter([median], [index], marker="D", s=54, color=INK, zorder=4)
        ax.text(median, index-0.27, f"{median:.3f}", ha="center", fontsize=11, color=INK, weight="bold")
    ax.set_yticks(range(4), [label for _, label in measures])
    ax.set_ylim(3.6, -0.7)
    ax.set_xlim(-0.4, 1.04)
    ax.set_xticks([-0.4, 0, 0.4, 0.8, 1.0])
    ax.axvline(0, color="#BBC5CC", lw=0.9)
    ax.set_xlabel("Metric value", labelpad=10)
    ax.grid(axis="x", color=PALE, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=12)
    rows = []
    labels = []
    for partition, part_label in (("development", "Development"), ("followup", "Follow-up")):
        for outcome, short in (("NDS_like_loss", "NDS-like loss"),
                               ("lost_clean_tp_fraction", "Lost-TP fraction"),
                               ("lost_clean_tp_count", "Lost-TP count")):
            summary = data[partition]["outcomes"][outcome]["fractions"]["0.20"]["decision_consequence"]["rules"]["raw_overlap"]["summary"]
            available = summary["mean_availability_regret"] / summary["mean_total_reuse_regret"]
            rows.append({"partition": partition, "loss": outcome,
                         "candidate_set_share": available, "source_choice_share": 1-available,
                         "mean_total_regret": summary["mean_total_reuse_regret"]})
            labels.append(part_label + "\n" + short)
    ax = axes[1]
    ax.set_title("B   Where target-oracle reuse regret arises", loc="left", pad=22, fontsize=12, weight="bold")
    for index, row in enumerate(rows):
        value = row["candidate_set_share"]
        ax.barh(index, value, height=0.58, color=BLUE)
        ax.barh(index, 1-value, left=value, height=0.58, color=GOLD)
        ax.text(value/2, index, f"{value:.1%}", ha="center", va="center", color="white", weight="bold", fontsize=11)
    ax.set_yticks(range(6), labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0,1)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xticks([0,0.25,0.5,0.75,1])
    ax.set_xlabel("Share of mean total regret", labelpad=10)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.axhline(2.5, color=PALE, lw=1.0)
    ax.legend([plt.Rectangle((0,0),1,1,color=BLUE),plt.Rectangle((0,0),1,1,color=GOLD)],
              ["Available source sets", "Source choice"], frameon=False,
              loc="upper left", bbox_to_anchor=(-0.1,-0.17), ncol=2, fontsize=10)
    for ax in axes:
        for side in ("top","right","left"):ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#B9C5CE")
    fig.text(0.045,0.105,
             "A  Each dot is a directed detector-pair / perturbation cell (72 cells); diamonds mark medians. Lost-clean-TP count, top-20% scene budget.",
             size=10, color=MUTED)
    fig.text(0.045,0.065,
             "B  Raw scene-overlap selection between two source sets. Development: 12 signed perturbations; follow-up: six signed rotations, separate logs.",
             size=10, color=MUTED)
    for extension in ("png","svg"):
        output_path = args.output/f"transfer.{extension}"
        fig.savefig(output_path, dpi=180)
        if extension == "svg":
            output_path.write_text("\n".join(
                line.rstrip() for line in output_path.read_text().splitlines()
            ) + "\n")
    plt.close(fig)
    with (args.output/"regret.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


if __name__ == "__main__":
    main()
