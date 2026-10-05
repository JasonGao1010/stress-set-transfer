#!/usr/bin/env python3
"""Draw the stress-set transfer method overview as PNG, outlined SVG, and PDF.

Schematic scene drawings, rankings, and loss marks illustrate the metric
definitions in METHOD.md.
Run from the repository root: python analysis/method_figure.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D


W, H = 1800, 1040
C = {
    "paper": "#FFFFFF", "ink": "#183142", "muted": "#63747D",
    "rule": "#DEE6E9", "pale": "#F5F8F9", "road": "#E5ECEF",
    "lane": "#FFFFFF", "source": "#28688F", "source_light": "#E5F0F6",
    "target": "#B76C43", "target_light": "#FAEFE7",
    "shared": "#4A827B", "shared_light": "#E8F2EF", "car": "#87989F",
}


def make_figure(output: Path) -> None:
    """Render a deterministic illustration with Times New Roman prose."""
    for weight in ("normal", "bold"):
        font_manager.findfont(
            font_manager.FontProperties(family="Times New Roman", weight=weight),
            fallback_to_default=False,
        )
    plt.rcParams.update({
        "font.family": "Times New Roman", "text.color": C["ink"],
        "mathtext.fontset": "stix", "svg.fonttype": "path",
        "svg.hashsalt": "stress-set-transfer-method",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": C["paper"],
    })
    fig = plt.figure(figsize=(18, 10.4), dpi=100, facecolor=C["paper"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(H, 0), aspect="equal")
    ax.axis("off")

    def label(x, y, value, size=17, color="ink", weight="normal", ha="left", **kw):
        return ax.text(x, y, value, fontsize=size, color=C.get(color, color),
                       weight=weight, ha=ha, va="center", **kw)

    def box(x, y, w, h, fill="paper", stroke="rule", radius=14, lw=1):
        shape = FancyBboxPatch((x, y), w, h,
                              boxstyle=f"round,pad=0,rounding_size={radius}",
                              facecolor=C.get(fill, fill), edgecolor=C.get(stroke, stroke),
                              linewidth=lw)
        ax.add_patch(shape)
        return shape

    def line(x1, y1, x2, y2, color="rule", lw=1.2, **kw):
        ax.plot([x1, x2], [y1, y2], color=C.get(color, color), lw=lw, **kw)

    def arrow(x1, y1, x2, y2, color="muted", scale=17, lw=1.8):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                    mutation_scale=scale, color=C.get(color, color),
                                    linewidth=lw, shrinkA=0, shrinkB=0))

    def dot(x, y, r, color, edge=None):
        ax.add_patch(Circle((x, y), r, facecolor=C.get(color, color),
                            edgecolor=C.get(edge, edge) if edge else "none", lw=1))

    def car(cx, cy, w=16, h=28, angle=0, color="car"):
        transform = Affine2D().rotate_deg_around(cx, cy, angle) + ax.transData
        body = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                              boxstyle="round,pad=0,rounding_size=3",
                              facecolor=C[color], edgecolor="white", lw=0.6,
                              transform=transform)
        ax.add_patch(body)
        ax.add_patch(Rectangle((cx-w*.31, cy-h*.27), w*.62, h*.17,
                               facecolor="#D4E1E6", edgecolor="none", transform=transform))

    def street(x, y, w, h, scene):
        clip = box(x, y, w, h, fill="pale", radius=10)
        # Simplified bird's-eye street geometry: three distinct scene contexts.
        def rect(rx, ry, rw, rh, fill):
            p = Rectangle((rx, ry), rw, rh, facecolor=C.get(fill, fill), edgecolor="none")
            p.set_clip_path(clip)
            ax.add_patch(p)
        roadx = x+w*.48
        rect(roadx-25, y, 50, h, "road")
        if scene in ("A", "C"):
            rect(x, y+h*.45-23, w, 46, "road")
            for lx in (x+8, x+w-23):
                line(lx, y+h*.45, lx+15, y+h*.45, "lane", 2)
        for dy in (10, h-23):
            line(roadx, y+dy, roadx, y+dy+12, "lane", 2)
        # Building footprints and small trees orient each drawing.
        for bx, by, bw, bh in ((x+9, y+11, 27, 34), (x+w-36, y+h-42, 26, 31)):
            rect(bx, by, bw, bh, "#D6E1E4")
            rect(bx+4, by+4, bw-8, bh-8, "#E5EDEF")
        dot(x+w-21, y+21, 7, "#BCD2C8")
        dot(x+20, y+h-19, 7, "#BCD2C8")
        ego_y = y+h-31
        # Translucent fan indicates the sensor's field of view.
        fan = Polygon([(roadx-11, ego_y), (roadx-42, ego_y-71),
                       (roadx+22, ego_y-71)], closed=True,
                      facecolor=C["source"], edgecolor="none", alpha=.10)
        fan.set_clip_path(clip)
        ax.add_patch(fan)
        car(roadx-11, ego_y, color="source")
        car(roadx+11, y+30, angle=180)
        if scene == "A":
            car(x+w-28, y+h*.45-10, angle=90)
            car(x+27, y+h*.45+10, angle=-90)
        elif scene == "B":
            car(roadx-11, y+48, h=35, w=19)
            car(roadx+11, y+83, angle=180)
        else:
            car(x+33, y+h*.45+10, angle=-90)
            dot(x+w-32, y+h*.45+32, 3.5, "target")
            dot(x+w-23, y+h*.45+32, 3.5, "target")
        box(x+8, y+h-27, 24, 21, fill="paper", stroke="paper", radius=4)
        label(x+20, y+h-16, scene, 11, weight="bold", ha="center")

    def scene_token(x, y, name, fill="source_light", stroke="source", w=51, h=49,
                    text_color=None, label_y=None):
        box(x, y, w, h, fill=fill, stroke=stroke, radius=8, lw=1.2)
        label(x+w/2, y+(h/2 if label_y is None else label_y), name, 18,
              color=text_color or stroke, weight="bold", ha="center")

    def step(x, n, title, subtitle):
        dot(x+18, 222, 18, "ink")
        label(x+18, 223, n, 14, color="paper", weight="bold", ha="center")
        label(x+48, 222, title, 22, weight="bold")
        label(x+48, 259, subtitle, 17, color="muted")

    # Header.
    label(65, 53, "STRESS-SET TRANSFER", 16, color="source", weight="bold")
    label(65, 109, "When do one detector’s hard scenes transfer to another?", 32, weight="bold")
    label(65, 154, "Source-selected scenes · target-specific losses · equal evaluation budgets", 18, color="muted")
    line(65, 185, 1735, 185)

    # A source and a target see the same physical scenes and conditions.
    step(65, "1", "A shared driving-scene pool", "Matched frames and calibration conditions")
    for x, name in zip((67, 208, 349), ("A", "B", "C")):
        street(x, 294, 125, 159, name)
    label(505, 379, "…", 25, color="muted", ha="center")
    label(289, 482, "Camera + LiDAR observations", 17, ha="center")
    label(289, 514, "Clean calibration and signed perturbations", 15, color="muted", ha="center")
    arrow(543, 375, 599, 375)

    # Source mining: illustration ranks differ from target rankings.
    step(630, "2", "Mine scenes with a source", "Select the k highest-loss scenes")
    box(636, 296, 191, 57, fill="source_light", stroke="source_light", radius=9)
    label(731.5, 324.5, "Source detector", 19, color="source", weight="bold", ha="center")
    label(667, 377, "Higher source loss", 14, color="muted")
    arrow(874, 377, 1004, 377, color="source", scale=13, lw=1.3)
    order = ["F", "D", "B", "E", "C", "A"]
    for i, name in enumerate(order):
        selected = i >= 3
        scene_token(637+i*64, 402, name,
                    fill="source_light" if selected else "pale",
                    stroke="source" if selected else "rule",
                    text_color="source" if selected else "muted")
    box(822, 392, 200, 69, fill="none", stroke="source", radius=13, lw=1.8)
    label(922, 490, r"Source stress set $S$", 18, color="source", weight="bold", ha="center")
    label(833, 525, "Retain the selected scene identities", 15, color="muted", ha="center")
    arrow(1053, 425, 1151, 425, color="source")
    label(1102, 387, "Reuse", 15, color="source", ha="center")

    # Same selected set, evaluated using target losses.
    step(1183, "3", "Evaluate on a target", "Score all scenes with the target detector")
    box(1190, 296, 191, 57, fill="target_light", stroke="target_light", radius=9)
    label(1285.5, 324.5, "Target detector", 19, color="target", weight="bold", ha="center")
    label(1194, 377, "Higher target loss", 14, color="muted")
    arrow(1440, 377, 1664, 377, color="target", scale=13, lw=1.3)
    for i, (name, count) in enumerate(zip(("F", "D", "C", "E", "B", "A"), (1, 1, 1, 2, 3, 4))):
        x = 1194+i*82
        selected = name in "ACE"
        scene_token(x, 402, name, w=60,
                    fill="source_light" if selected else "pale",
                    stroke="source" if selected else "rule",
                    text_color="source" if selected else "muted")
        for j in range(4):
            dot(x+13+j*11, 477, 4, "target" if j < count else "target_light")
    box(1430, 392, 244, 69, fill="none", stroke="target", radius=13, lw=1.8)
    label(1431, 514, r"Blue: reused $S$   ·   Orange outline: target top-$k$", 16, ha="center")
    # A restrained bracket links the pipeline to all three readouts.
    line(65, 562, 1735, 562)
    label(65, 599, "WHAT DOES REUSE PRESERVE?", 15, color="muted", weight="bold")
    label(1735, 599, "Schematic scenes, rankings, and loss marks", 13, color="muted", ha="right")

    # Three metric cards. Their graphical encodings explain the definitions;
    # their sizes and colors are not observed study values.
    card_y, card_h, card_w = 631, 315, 530
    for x in (65, 635, 1205):
        box(x, card_y, card_w, card_h, fill="pale", stroke="pale", radius=16)

    label(91, 668, "Scene identity agreement", 21, weight="bold")
    label(91, 702, "Do the selected hard scenes coincide?", 17, color="muted")
    # Explicit set membership instead of an area-proportional Venn diagram.
    label(92, 758, "Source S", 16, color="source")
    label(92, 817, "Target T", 16, color="target")
    for i, name in enumerate(("E", "C", "A")):
        scene_token(249+i*91, 734, name, w=63, h=47,
                    fill="shared_light" if name != "C" else "source_light",
                    stroke="shared" if name != "C" else "source")
    for i, name in enumerate(("E", "B", "A")):
        scene_token(249+i*91, 794, name, w=63, h=47,
                    fill="shared_light" if name != "B" else "target_light",
                    stroke="shared" if name != "B" else "target")
    label(330, 872, r"$|S\cap T|\,/\,k$", 25, ha="center")
    label(330, 922, "Shared identities / selection budget", 14, color="muted", ha="center")

    label(661, 668, "Target-loss coverage", 21, weight="bold")
    label(661, 702, "How much target loss is captured?", 17, color="muted")
    # Loss contributions from all scenes; outlined cells are source-selected.
    for i, (name, count) in enumerate(zip("ABCDEF", (4, 3, 1, 1, 2, 1))):
        x = 670+i*80
        selected = name in "ACE"
        scene_token(x, 742, name, w=61, h=78, label_y=20,
                    fill="source_light" if selected else "paper",
                    stroke="source" if selected else "rule",
                    text_color="source" if selected else "muted")
        for j in range(count):
            dot(x+17+(j%2)*25, 788+(j//2)*14, 4, "target")
    label(900, 872, r"$\sum_{s\in S}\ell_t(s)\,/\,\sum_{s\in\mathcal{D}}\ell_t(s)$", 23, ha="center")
    label(900, 922, "Captured target loss / loss across all scenes", 14, color="muted", ha="center")

    label(1231, 668, "Same-budget efficiency", 21, weight="bold")
    label(1231, 702, "How useful is the reused selection?", 17, color="muted")
    # Equal scene counts make the target-oracle comparison explicit.
    label(1232, 758, "Reuse S", 16, color="source")
    label(1232, 817, "Target top-k", 16, color="target")
    for row_y, names, col, fill in ((734, ("E", "C", "A"), "source", "source_light"),
                                  (794, ("E", "B", "A"), "target", "target_light")):
        for i, name in enumerate(names):
            scene_token(1387+i*91, row_y, name, fill=fill, stroke=col, w=63, h=47)
    label(1470, 872, r"$\sum_{s\in S}\ell_t(s)\,/\,\sum_{s\in T}\ell_t(s)$", 23, ha="center")
    label(1470, 922, "Captured loss / target top-k loss", 14, color="muted", ha="center")

    # Verified study scope, separated from the illustrative panel.
    label(65, 982, "BEVFusion · SparseFusion · DeepInteraction", 16, weight="bold")
    label(1735, 982, "nuScenes  |  50 development + 18 follow-up scenes", 16, color="muted", ha="right")
    label(65, 1016, r"$\ell_t(s)$: target loss in scene $s$     $\mathcal{D}$: all evaluated scenes     $T$: target top-$k$ set", 13, color="muted")
    label(1735, 1016, "Exact tie handling · acquisition-log and opportunity references", 13, color="muted", ha="right")

    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"method.{extension}", dpi=160,
                    metadata={"Creator": "", "CreationDate": None, "ModDate": None}
                    if extension == "pdf" else {"Creator": "", "Date": None}
                    if extension == "svg" else None)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "assets")
    args = parser.parse_args()
    make_figure(args.output)


if __name__ == "__main__":
    main()
