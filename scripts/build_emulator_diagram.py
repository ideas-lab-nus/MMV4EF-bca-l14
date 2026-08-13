"""Build the manuscript's mode-specific thermal-emulator schematic."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures" / "emulator_diagram.pdf"


COLORS = {
    "state": "#E8F0F7",
    "ac": "#EAF4EC",
    "nv": "#FFF1E5",
    "network": "#F2EDF8",
    "result": "#FFF7D6",
    "line": "#34495E",
    "text": "#17202A",
}


def box(ax, x, y, w, h, text, *, facecolor, fontsize=9.2, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.014",
        linewidth=1.15,
        edgecolor=COLORS["line"],
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=COLORS["text"],
        linespacing=1.18,
        zorder=3,
    )
    return patch


def arrow(ax, start, end, *, connectionstyle="arc3", linewidth=1.25):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=linewidth,
        color=COLORS["line"],
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
        zorder=1,
    )
    ax.add_patch(patch)
    return patch


def branch(ax, y, *, mode, controls, exogenous, input_dim, output_label, color):
    lane = FancyBboxPatch(
        (0.025, y - 0.035),
        0.735,
        0.285,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.15,
        edgecolor=color,
        facecolor=color,
        alpha=0.16,
        zorder=0,
    )
    ax.add_patch(lane)
    ax.text(0.045, y + 0.218, mode, fontsize=8.9, fontweight="bold", color=COLORS["text"])

    box(ax, 0.045, y + 0.095, 0.145, 0.095, controls, facecolor=color, fontsize=7.8)
    box(ax, 0.045, y - 0.005, 0.145, 0.075, exogenous, facecolor=color, fontsize=7.5)
    box(
        ax,
        0.225,
        y + 0.025,
        0.135,
        0.14,
        "Concatenated\ninput\n"
        + r"$[x_t;u_t;w_t]$"
        + "\n"
        + rf"$\in \mathbb{{R}}^{{{input_dim}}}$",
        facecolor="white",
        fontsize=6.7,
    )
    box(ax, 0.39, y + 0.035, 0.095, 0.11, "Conv1D\n15 → 64" if input_dim == 15 else "Conv1D\n12 → 64", facecolor=COLORS["network"], fontsize=7.5)
    box(ax, 0.515, y + 0.035, 0.095, 0.11, "LSTM\n$h=64$", facecolor=COLORS["network"], fontsize=7.7)
    box(ax, 0.64, y + 0.035, 0.09, 0.11, "FC\n64 → 5", facecolor=COLORS["network"], fontsize=7.5)

    arrow(ax, (0.19, y + 0.142), (0.225, y + 0.105))
    arrow(ax, (0.19, y + 0.033), (0.225, y + 0.073))
    arrow(ax, (0.36, y + 0.09), (0.39, y + 0.09))
    arrow(ax, (0.485, y + 0.09), (0.515, y + 0.09))
    arrow(ax, (0.61, y + 0.09), (0.64, y + 0.09))
    ax.text(0.74, y + 0.09, output_label, ha="left", va="center", fontsize=8.6, fontweight="bold")


def build(output: Path = OUTPUT) -> Path:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.25, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(
        ax,
        0.04,
        0.875,
        0.17,
        0.085,
        "Current state\n$x_t ∈ \mathbb{R}^5$",
        facecolor=COLORS["state"],
        fontsize=8.5,
        weight="bold",
    )

    branch(
        ax,
        0.555,
        mode="Air-conditioning (AC) branch",
        controls="AC controls\n$u_t^{ac} ∈ \mathbb{R}^7$",
        exogenous="Disturbances\n$w_t^{ac} ∈ \mathbb{R}^3$",
        input_dim=15,
        output_label="$f_{ac}$",
        color=COLORS["ac"],
    )
    branch(
        ax,
        0.205,
        mode="Natural-ventilation (NV) branch",
        controls="NV controls\n$u_t^{nv} ∈ \mathbb{R}^4$",
        exogenous="Disturbances\n$w_t^{nv} ∈ \mathbb{R}^3$",
        input_dim=12,
        output_label="$f_{nv}$",
        color=COLORS["nv"],
    )

    box(
        ax,
        0.805,
        0.49,
        0.16,
        0.13,
        "Mode-selected\nstate transition\n$f_t$",
        facecolor=COLORS["result"],
        fontsize=7.2,
        weight="bold",
    )
    arrow(ax, (0.77, 0.645), (0.805, 0.575), connectionstyle="arc3,rad=0.12")
    arrow(ax, (0.77, 0.295), (0.805, 0.535), connectionstyle="arc3,rad=-0.12")

    box(
        ax,
        0.805,
        0.715,
        0.16,
        0.105,
        "Closed-loop update\n$x_{t+1}=f_t(x_t,u_t,w_t)$",
        facecolor=COLORS["state"],
        fontsize=6.8,
    )
    arrow(ax, (0.885, 0.62), (0.885, 0.715))

    box(
        ax,
        0.805,
        0.175,
        0.16,
        0.15,
        "Instantaneous\nlinearization\n$\hat{x}_{t+1}=A_t x_t$\n$+B_t u_t+c_t$",
        facecolor=COLORS["network"],
        fontsize=7.0,
    )
    arrow(ax, (0.885, 0.49), (0.885, 0.325))
    # Receding-horizon feedback path.
    arrow(ax, (0.805, 0.77), (0.215, 0.935), connectionstyle="arc3,rad=0.18", linewidth=1.1)
    ax.text(0.54, 0.965, "next control interval", ha="center", va="bottom", fontsize=7.4, color=COLORS["line"])

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return output


if __name__ == "__main__":
    print(build())
