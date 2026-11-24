"""Plot the saved experimental results for the project report."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]


def configure_font():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    families = ("Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "STHeiti", "PingFang HK")
    font = next((name for name in families if name in installed), None)
    if font is None:
        raise RuntimeError("Install Noto Sans CJK SC to render the Chinese labels")
    plt.rcParams.update({"font.family": font, "font.size": 10.5, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .7, "legend.frameon": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})


def main():
    configure_font()
    table = pd.read_csv(ROOT / "manufacturing/results/station_five_study.csv")
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8.5, 3.7), layout="constrained")
    for ax, (pressure, title) in zip(axes, (("reference", "原设定"), ("combined", "组合压力")), strict=True):
        group = table.loc[table.pressure == pressure]
        # Select using the baseline only; keep both observations from the same trajectory.
        top = group.nlargest(40, "old_peak").sort_values(["old_peak", "trajectory"])
        rank = np.arange(1, len(top)+1)
        ax.vlines(rank, top.new_peak, top.old_peak, color=".80", lw=.7)
        ax.scatter(rank, top.old_peak, marker="o", color="#476b83", s=15, label="原动作")
        ax.scatter(rank, top.new_peak, marker="^", color="#a45735", s=16, label="B 型件减一次动作")
        ax.axhline(42, color=".35", lw=.9, ls="--", label="设定上限")
        ax.set(title=title, xticks=[1, 10, 20, 30, 40], xlabel="高负荷轨迹序号", ylim=(37.1, 42.65))
    axes[0].set_ylabel("全线最高班平均动作频次（次/分钟）")
    axes[0].legend(loc="lower left", fontsize=8, ncol=2, handlelength=1.3,
                   columnspacing=.9, labelspacing=.3)
    fig.savefig(ROOT / "images/station_five.png", dpi=200)

    plt.close(fig)


if __name__ == "__main__":
    main()
