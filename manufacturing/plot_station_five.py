"""Empirical peak action rates before and after a station-five change."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from factory_floor_study import ACTIONS_LIMIT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figure_style import format_figure  # noqa: E402


def main():
    data = pd.read_csv(ROOT / "manufacturing/results/station_five_study.csv")
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3.45), layout="constrained")
    for ax, (pressure, label) in zip(axes, [("reference", "A    原设定"), ("combined", "B    组合压力")]):
        rows = data.loc[data.pressure == pressure]
        for field, name, color, style in [("old_peak", "原分工", "#5c6165", "--"),
                                           ("new_peak", "调整后", "#224f6d", "-")]:
            x = np.sort(rows[field])
            y = np.arange(1, x.size+1) * 100 / x.size
            ax.step(np.r_[29, x, 45], np.r_[0, y, 100], where="post", color=color, ls=style, lw=1.5, label=name)
        ax.axvline(ACTIONS_LIMIT, color="#98633f", lw=.9, ls=":", label="设定上限")
        ax.set_title(label, loc="left")
        ax.set(xlim=(29,45), ylim=(-3,106), xticks=[30,33,36,39,42,45])
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("累计轨迹占比（%）")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.supxlabel("最高班平均动作频次（次/分钟）", fontsize=11)
    format_figure(fig)
    fig.savefig(ROOT / "images/station_five.png", dpi=260)
    plt.close(fig)


if __name__ == "__main__":
    main()
