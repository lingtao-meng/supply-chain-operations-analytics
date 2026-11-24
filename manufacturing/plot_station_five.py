"""Compare original and adjusted peak action rates for every simulated trajectory."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

from factory_floor_study import ACTIONS_LIMIT

ROOT = Path(__file__).resolve().parents[1]
INK, GRAY = "#1b2c38", "#4e606d"
BLUE, ORANGE = "#236b8e", "#a64b30"



def configure_style():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ("Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "STHeiti", "PingFang HK")
    family = next((name for name in candidates if name in installed), None)
    if family is None:
        raise RuntimeError("Install Noto Sans CJK SC to render the Chinese labels")
    plt.rcParams.update({
        "font.family": family, "font.size": 11, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": "#aebac2", "axes.linewidth": .75,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": GRAY, "ytick.color": GRAY, "axes.unicode_minus": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "legend.frameon": False, "grid.color": "#e5eaee", "grid.linewidth": .7,
    })


def main():
    configure_style()
    table = pd.read_csv(ROOT / "manufacturing/results/station_five_study.csv")
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8.65, 4.9))
    fig.subplots_adjust(left=.105, right=.965, top=.875, bottom=.19, wspace=.18)
    limits = (30, 44)
    ticks = (30, 33, 36, 39, 42)
    for ax, (pressure, label) in zip(axes, (("reference", "原设定"), ("combined", "组合压力")), strict=True):
        group = table.loc[table.pressure == pressure]
        before = group.old_peak.gt(ACTIONS_LIMIT)
        after = group.new_peak.gt(ACTIONS_LIMIT)
        ax.axvspan(ACTIONS_LIMIT, limits[1], facecolor="#faefea", linewidth=0)
        ax.axhspan(ACTIONS_LIMIT, limits[1], facecolor="#faefea", linewidth=0)
        ax.plot(limits, limits, linestyle=(0, (2, 3)), color="#798895", linewidth=.9)
        ax.axvline(ACTIONS_LIMIT, color=ORANGE, linewidth=.95, linestyle="--")
        ax.axhline(ACTIONS_LIMIT, color=ORANGE, linewidth=.95, linestyle="--")
        ax.scatter(group.old_peak, group.new_peak, color=BLUE, s=15, alpha=.25, linewidths=0, zorder=3)
        flagged = group.loc[before | after]
        ax.scatter(flagged.old_peak, flagged.new_peak, marker="o", facecolors="white",
                   edgecolors=ORANGE, s=50, linewidths=1.3, zorder=4)
        ax.set(xlim=limits, ylim=limits, xticks=ticks, yticks=ticks, aspect="equal")
        ax.set_title(label, loc="left", pad=9)
        ax.text(.06, .81, f"动作频次超限\n{before.sum()}/{len(group)} → {after.sum()}/{len(group)}",
                transform=ax.transAxes, va="top", fontsize=10, linespacing=1.45)
        ax.tick_params(length=2.5)
    fig.supxlabel("调整前最高班平均动作频次（次/分钟）", y=.070, fontsize=11, color=INK)
    fig.supylabel("调整后最高班平均动作频次（次/分钟）", x=.025, fontsize=11, color=INK)
    fig.savefig(ROOT / "images/station_five.png", dpi=210)
    plt.close(fig)


if __name__ == "__main__":
    main()
