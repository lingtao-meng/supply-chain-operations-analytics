"""Plot cross-shift rework backlog and due-shift delivery from the saved experiment."""

import argparse
import json
from pathlib import Path

import matplotlib

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_style import format_figure  # noqa: E402


matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt

ROOT = Path(__file__).resolve().parent
POLICIES = (
    ("no_support", "无支援", "#534b47", "o"),
    ("one_90", "1 人 × 90 分钟", "#876c44", "s"),
    ("one_240", "1 人 × 240 分钟", "#3a5875", "D"),
    ("two_120", "2 人各 120 分钟", "#3c6866", "^"),
)


def choose_font(path: Path | None) -> str:
    if path:
        font_manager.fontManager.addfont(path)
        return font_manager.FontProperties(fname=path).get_name()
    for candidate in ("Noto Serif CJK SC", "Songti SC", "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", "Heiti SC", "Heiti TC"):
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
        except ValueError:
            continue
        return candidate
    raise ValueError("请安装中文字体，或用 --font 指定中文字体文件")


def write_chart(report: dict, path: Path, family: str) -> None:
    start = report["phases"].index("adverse") + 1
    end = report["phases"].index("recovery")
    shifts = range(1, len(report["phases"]) + 1)
    with plt.rc_context({"font.family": family, "axes.unicode_minus": False, "font.size": 11}):
        fig, (left, right) = plt.subplots(1, 2, figsize=(10.2, 4.1), layout="constrained")
        for ax in (left, right):
            ax.axvspan(start - .5, end + .5, color="#eef0f0", zorder=0)
            ax.grid(axis="y", color="#DDE4E9", linewidth=.7)
            ax.set_axisbelow(True)
            ax.set_xticks(shifts)
            ax.set_xlim(.8, len(report["phases"]) + .2)
            ax.set_xlabel("班次")
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["bottom", "left"]].set_color("#B0BBC4")
            ax.tick_params(length=0, pad=6, colors="#34414A")

        for name, label, color, marker in POLICIES:
            points = report["summary"][name]["by_shift"]
            left.plot(shifts, [row["mean_pending_rework"] for row in points], label=label,
                      color=color, marker=marker, markersize=4, linewidth=2)
        left.set_title("A    班末平均待返工量", loc="left")
        left.set_ylabel("件")
        left.set_ylim(0)
        left.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)

        baseline = report["summary"]["no_support"]["by_shift"]
        for key, label, color, style in (
            ("histories_at_physical_target", "当班总放行达标", "#3a5875", "--"),
            ("histories_at_due_target", "本班应交件按时达标", "#AE5035", "-"),
        ):
            values = [100 * row[key] / report["histories"] for row in baseline]
            right.plot(shifts, values, label=label, color=color, linestyle=style, marker="o",
                       markersize=4, linewidth=2)
        right.set_title("B    无支援时的班产与按期交付", loc="left")
        right.set_ylabel("达到 1,650 件的仿真比例（%）")
        right.set_ylim(0, 110)
        right.legend(loc="lower left", fontsize=9, frameon=False)
        right.text(.98, .06, f"底色表示第 {start}–{end} 班为异常工况", transform=right.transAxes,
                   ha="right", fontsize=9, color="#65584D")
        path.parent.mkdir(parents=True, exist_ok=True)
        format_figure(fig, fonts=["STIXGeneral", family])
        fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "rework_carryover.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "rework_carryover.png")
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    write_chart(json.loads(args.input.read_text(encoding="utf-8")), args.output, choose_font(args.font))


if __name__ == "__main__":
    main()
