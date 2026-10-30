#!/usr/bin/env python3
"""Sorting-line scenarios: line balance, queue simulation, staffing and cost.

The time-study numbers in ``data/time_study.csv`` are a worked example.  Replace
them with your own stopwatch data and the same pipeline produces the Yamazumi
chart, balance rate, bottleneck, overtime and cost per parcel.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hub import line_metrics, load_stations, simulate, with_operator  # noqa: E402

RESULTS = ROOT / "results"
IMAGES = ROOT / "images"
INK = "#2b2b2b"
TEAL = "#1f6f78"
AMBER = "#c9a227"
RED = "#b23a48"
MUTED = "#9aa6ad"
GRID = "#e8e8e8"

REPLICATIONS = 30
GRID_REPLICATIONS = 5
GRID_OPERATORS = (1, 2, 3)
SCENARIOS = [
    ("base", 220.0, None),
    ("peak", 286.0, None),
    ("peak_improved", 286.0, ("induct", 2)),
    ("base_improved", 220.0, ("induct", 2)),
]


def _style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "#fbfbf8",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#d9d9d9",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
        }
    )


def plot_yamazumi(base, improved, metrics_base, metrics_improved, images: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    for ax, stations, metrics, title in [
        (axes[0], base, metrics_base, "before"),
        (axes[1], improved, metrics_improved, "after +1 induct operator"),
    ]:
        names = [station.name for station in stations]
        cycles = [station.effective_cycle for station in stations]
        colors = [RED if name == metrics.bottleneck else TEAL for name in names]
        bars = ax.bar(names, cycles, color=colors, width=0.58)
        for bar, value in zip(bars, cycles):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.2, f"{value:.1f}s", ha="center", fontsize=8)
        ax.axhline(metrics.bottleneck_cycle, color=RED, linestyle="--", linewidth=1)
        ax.set_title(f"{title}  ·  balance {metrics.balance_rate:.0%}", loc="left")
        ax.set_ylim(0, max(cycles) * 1.2)
        ax.set_ylabel("effective cycle time per operator (s)")
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(images / "yamazumi.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_scenarios(table: pd.DataFrame, images: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    labels = ["base", "peak", "peak\n+1 induct", "base\n+1 induct"]
    colors = [MUTED, RED, TEAL, AMBER]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))
    panels = [
        ("on_time_rate", "on-time within 5 min (%)", 100.0),
        ("avg_flow_min", "average flow time (min)", 1.0),
        ("overtime_min", "overtime (min)", 1.0),
        ("cost_per_parcel", "cost per parcel (USD)", 1.0),
    ]
    for ax, (column, title, scale) in zip(axes.flat, panels):
        values = table[column].to_numpy() * scale
        bars = ax.bar(labels, values, color=colors, width=0.58)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.02,
                    f"{value:.1f}" if column != "on_time_rate" else f"{value:.0f}%",
                    ha="center", fontsize=8)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, max(values) * 1.18)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(images / "scenarios.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_backlog(results: dict, images: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for name, color in [("base", MUTED), ("peak", RED), ("peak_improved", TEAL)]:
        backlog = results[name].backlog
        ax.plot(np.arange(len(backlog)), backlog, color=color, linewidth=1.8, label=name)
    ax.axvline(480, color="#6b6b6b", linestyle=":", linewidth=1)
    ax.text(482, ax.get_ylim()[1] * 0.92, "shift end", fontsize=8, color="#5f6b70")
    ax.set_xlabel("minute")
    ax.set_ylabel("parcels waiting in the line")
    ax.set_title("Backlog: the peak hour is what breaks the line", loc="left")
    ax.grid(axis="y")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(images / "backlog.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _summarise(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, mean - half, mean + half


def staffing_grid(base, rate: float, replications: int, seed: int) -> pd.DataFrame:
    rows = []
    for combo in product(GRID_OPERATORS, repeat=len(base)):
        stations = [replace(station, operators=n) for station, n in zip(base, combo)]
        metrics = line_metrics(stations)
        runs = [simulate(stations, rate, seed=seed + i) for i in range(replications)]
        cost = np.array([r.cost_per_parcel for r in runs])
        on_time = np.array([r.on_time_rate for r in runs])
        rows.append({
            "config": "-".join(str(n) for n in combo),
            "operators": sum(combo),
            "balance_rate": metrics.balance_rate,
            "line_capacity_per_hour": metrics.capacity_per_hour,
            "bottleneck": metrics.bottleneck,
            "cost_per_parcel": float(cost.mean()),
            "cost_ci_low": float(cost.mean() - 1.96 * cost.std(ddof=1) / np.sqrt(len(cost))),
            "cost_ci_high": float(cost.mean() + 1.96 * cost.std(ddof=1) / np.sqrt(len(cost))),
            "on_time_rate": float(on_time.mean()),
            "on_time_ci_low": float(on_time.mean() - 1.96 * on_time.std(ddof=1) / np.sqrt(len(on_time))),
            "on_time_ci_high": float(on_time.mean() + 1.96 * on_time.std(ddof=1) / np.sqrt(len(on_time))),
            "avg_flow_min": float(np.mean([r.avg_flow_min for r in runs])),
            "overtime_min": float(np.mean([r.overtime_min for r in runs])),
        })
    return pd.DataFrame(rows)


def pareto_frontier(grid: pd.DataFrame) -> pd.DataFrame:
    out = []
    best_on_time = -1.0
    for _, row in grid.sort_values(["cost_per_parcel", "on_time_rate"], ascending=[True, False]).iterrows():
        if row["on_time_rate"] > best_on_time + 1e-12:
            out.append(row)
            best_on_time = float(row["on_time_rate"])
    return pd.DataFrame(out)


def plot_pareto(grid: pd.DataFrame, frontier: pd.DataFrame, images: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style()
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.scatter(grid["cost_per_parcel"], grid["on_time_rate"] * 100, s=18, color=MUTED, alpha=0.55, label="243 staffing plans")
    ax.plot(frontier["cost_per_parcel"], frontier["on_time_rate"] * 100, color=TEAL, marker="o", linewidth=2, label="Pareto frontier")
    for _, row in frontier.iterrows():
        ax.annotate(row["config"], (row["cost_per_parcel"], row["on_time_rate"] * 100), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("cost per parcel (USD)")
    ax.set_ylabel("on-time within 5 min (%)")
    ax.set_title("Staffing plans: cost vs service (peak 286/h)", loc="left")
    ax.grid(axis="y")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(images / "staffing_pareto.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--time-study", default=str(ROOT / "data" / "time_study.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)

    base = load_stations(Path(args.time_study))
    improved = with_operator(base, "induct", 2)
    results, rows, balance_rows = {}, [], []
    for name, rate, change in SCENARIOS:
        stations = base if change is None else with_operator(base, change[0], change[1])
        metrics = line_metrics(stations)
        runs = [simulate(stations, rate, seed=args.seed + i) for i in range(REPLICATIONS)]
        sim = runs[0]
        results[name] = sim
        row = {"scenario": name, "arrival_per_hour": rate, "operators": sim.total_operators,
               "bottleneck": metrics.bottleneck, "balance_rate": metrics.balance_rate,
               "line_capacity_per_hour": metrics.capacity_per_hour, "replications": REPLICATIONS}
        for key in ("throughput_per_hour", "avg_flow_min", "p95_flow_min", "on_time_rate", "overtime_min", "cost_per_parcel"):
            mean, lo, hi = _summarise([getattr(r, key) for r in runs])
            row[key] = mean
            row[f"{key}_ci_low"] = lo
            row[f"{key}_ci_high"] = hi
        rows.append(row)
        for station, cycle in zip(stations, metrics.effective_cycles):
            balance_rows.append(
                {
                    "scenario": name,
                    "station": station.name,
                    "work_content_sec": station.work_content,
                    "operators": station.operators,
                    "effective_cycle_sec": cycle,
                    "capacity_per_hour": station.capacity_per_hour,
                    "bottleneck": station.name == metrics.bottleneck,
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "scenarios.csv", index=False)
    pd.DataFrame(balance_rows).to_csv(RESULTS / "line_balance.csv", index=False)

    summary = {
        "assumptions": {
            "shift_hours": 8,
            "service_target_min": 5.0,
            "labor_rate_per_hour": 25.0,
            "equipment_per_parcel": 0.10,
            "time_study": str(Path(args.time_study)),
        },
        "scenarios": table.to_dict(orient="records"),
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )

    print(table.to_string(index=False))

    grid = staffing_grid(base, 286.0, GRID_REPLICATIONS, args.seed)
    grid.to_csv(RESULTS / "staffing_grid.csv", index=False)
    frontier = pareto_frontier(grid)
    frontier.to_csv(RESULTS / "staffing_pareto.csv", index=False)
    summary["staffing_grid"] = {
        "configs": len(grid),
        "replications": GRID_REPLICATIONS,
        "best_cost_meeting_90pct": float(grid[grid["on_time_rate"] >= 0.90]["cost_per_parcel"].min()) if (grid["on_time_rate"] >= 0.90).any() else None,
        "pareto_configs": frontier["config"].tolist(),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"staffing grid: {len(grid)} configs, pareto {len(frontier)}")

    metrics_base = line_metrics(base)
    metrics_improved = line_metrics(improved)
    plot_pareto(grid, frontier, IMAGES)
    plot_yamazumi(base, improved, metrics_base, metrics_improved, IMAGES)
    plot_scenarios(table, IMAGES)
    plot_backlog(results, IMAGES)
    print(f"wrote charts to {IMAGES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
