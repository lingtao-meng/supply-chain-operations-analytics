#!/usr/bin/env python3
"""IE scheduling simulation: dispatching rules, WIP and staffing.

The dashboard says *what* the delivery performance is; this script asks what a
planner can do about it under a finite daily capacity:

* orders arrive with a release day, a due day (order date + promised days) and
  a processing requirement (units);
* every day the warehouse can process ``capacity`` units;
* four dispatching rules are compared — FCFS, EDD, SPT, critical ratio (CR);
* KPIs: on-time rate, mean/max tardiness, flow time, WIP, utilisation;
* a staffing curve translates daily volume into required workers.

This is a deterministic discrete-event simulation on the last 15% of orders
(the same hold-out window as the ML project), not a forecast.

Usage
-----
    python python/04_ie_scheduling.py
    python python/04_ie_scheduling.py --capacity 200 --test-frac 0.2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
IMAGES = ROOT / "images"

from ie_common import DATE_COL, SCHED_COL, find_data, load_orders  # noqa: E402


@dataclass(frozen=True)
class Order:
    idx: int
    release: int
    due: int
    units: int
    mode: str


RULES = ("FCFS", "EDD", "SPT", "CR")


def rule_key(order: Order, day: int, rule: str):
    if rule == "FCFS":
        return (order.release, order.idx)
    if rule == "EDD":
        return (order.due, order.release, order.idx)
    if rule == "SPT":
        return (order.units, order.due, order.idx)
    if rule == "CR":  # critical ratio: small = urgent relative to work content
        return ((order.due - day) / max(order.units, 1), order.due, order.idx)
    raise ValueError(f"unknown rule: {rule}")


def build_orders(frame: pd.DataFrame) -> tuple[list[Order], pd.Timestamp]:
    frame = frame.sort_values(DATE_COL).reset_index(drop=True)
    origin = frame[DATE_COL].min().normalize()
    orders = []
    for idx, row in frame.iterrows():
        release = int((row[DATE_COL].normalize() - origin).days)
        due = release + int(row[SCHED_COL])
        orders.append(
            Order(
                idx=idx,
                release=release,
                due=due,
                units=int(row["Order Item Quantity"]),
                mode=str(row["Shipping Mode"]),
            )
        )
    return orders, origin


def simulate(orders: list[Order], capacity: int, rule: str, max_extra_days: int = 365) -> dict:
    """Single-stage dispatch with a daily unit capacity."""
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    by_release: dict[int, list[Order]] = {}
    for order in orders:
        by_release.setdefault(order.release, []).append(order)

    last_due = max(order.due for order in orders)
    queue: list[Order] = []
    completed: dict[int, int] = {}
    daily = []
    day = min(by_release)
    end_day = last_due + max_extra_days

    while day <= end_day:
        queue.extend(by_release.get(day, []))
        remaining = capacity
        queue.sort(key=lambda order: rule_key(order, day, rule))
        keep = []
        processed_units = 0
        for order in queue:
            if order.units <= remaining:
                completed[order.idx] = day
                remaining -= order.units
                processed_units += order.units
            else:
                keep.append(order)
        queue = keep
        daily.append(
            {
                "day": day,
                "processed_units": processed_units,
                "wip_orders": len(queue),
                "backlog_units": sum(order.units for order in queue),
            }
        )
        if not queue and day >= last_due:
            break
        day += 1

    incomplete = [order for order in orders if order.idx not in completed]
    for order in incomplete:  # pragma: no cover - should not happen with a sane horizon
        completed[order.idx] = end_day

    on_time = np.array([completed[o.idx] <= o.due for o in orders])
    tardiness = np.array([max(0, completed[o.idx] - o.due) for o in orders])
    flow = np.array([completed[o.idx] - o.release for o in orders])
    daily_df = pd.DataFrame(daily)
    active = daily_df[daily_df["processed_units"] > 0] if len(daily_df) else daily_df
    return {
        "rule": rule,
        "orders": len(orders),
        "on_time_rate": float(on_time.mean()),
        "avg_tardiness": float(tardiness.mean()),
        "late_avg_tardiness": float(tardiness[tardiness > 0].mean()) if (tardiness > 0).any() else 0.0,
        "max_tardiness": int(tardiness.max()),
        "avg_flow_days": float(flow.mean()),
        "avg_wip_orders": float(daily_df["wip_orders"].mean()),
        "utilization": float(active["processed_units"].sum() / (capacity * max(len(active), 1))),
        "incomplete": int(len(incomplete)),
        "daily": daily_df,
    }


def staffing_curve(frame: pd.DataFrame, units_per_worker: int = 40, target_utilization: float = 0.85) -> pd.DataFrame:
    """Workers needed per day after a 7-day rolling volume smoothing."""
    daily = (
        frame.assign(day=frame[DATE_COL].dt.normalize())
        .groupby("day")["Order Item Quantity"]
        .sum()
        .sort_index()
    )
    smoothed = daily.rolling(7, min_periods=1).mean()
    workers = np.ceil(smoothed / (units_per_worker * target_utilization))
    return pd.DataFrame({"day": daily.index, "units": daily.to_numpy(), "rolling_units": smoothed.to_numpy(), "workers_needed": workers.astype(int)})


def plot(results: dict[str, dict], staffing: pd.DataFrame, images: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 4.4))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.15], wspace=0.32)

    ax = fig.add_subplot(grid[0, 0])
    names = list(results)
    on_time = [results[name]["on_time_rate"] * 100 for name in names]
    tardy = [results[name]["avg_tardiness"] for name in names]
    x = np.arange(len(names))
    ax.bar(x - 0.2, on_time, width=0.4, color="#1f6f78", label="on-time %")
    ax.set_xticks(x, names)
    ax.set_ylabel("on-time rate (%)")
    ax.set_ylim(0, 105)
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, tardy, width=0.4, color="#d99a2b", label="avg tardiness (days)")
    ax2.set_ylabel("avg tardiness (days)")
    ax.set_title("Dispatching rules under a daily capacity")
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)

    ax3 = fig.add_subplot(grid[0, 1])
    for name, result in results.items():
        daily = result["daily"]
        ax3.plot(daily["day"], daily["wip_orders"], label=name, linewidth=1.2)
    ax3.set_xlabel("day")
    ax3.set_ylabel("WIP (orders in queue)")
    ax3.set_title("Backlog over time")
    ax3.legend(fontsize=7)

    ax4 = fig.add_subplot(grid[0, 2])
    monthly = staffing.set_index("day").resample("M").agg({"units": "mean", "workers_needed": "max"})
    ax4.bar(monthly.index, monthly["units"], width=20, color="#2f5d8c", alpha=0.65, label="daily units (avg)")
    ax4.set_ylabel("units / day")
    ax4.set_xlabel("month")
    ax4.tick_params(axis="x", rotation=45, labelsize=7)
    ax5 = ax4.twinx()
    ax5.plot(monthly.index, monthly["workers_needed"], color="#9b3b3b", marker="o", markersize=3, label="workers (max)")
    ax5.set_ylabel("workers needed")
    ax4.set_title("Staffing requirement (85% utilisation)")
    ax4.legend(loc="upper left", fontsize=7)
    ax5.legend(loc="upper right", fontsize=7)

    fig.savefig(images / "ie_scheduling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None)
    ap.add_argument("--capacity", type=int, default=None, help="units processed per day")
    ap.add_argument("--test-frac", type=float, default=0.15, help="last fraction of orders to simulate")
    ap.add_argument("--units-per-worker", type=int, default=40)
    ap.add_argument("--target-utilization", type=float, default=0.85)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)

    df = load_orders(find_data(args.data))
    start = int(len(df) * (1 - args.test_frac))
    test = df.iloc[start:].copy()
    orders, origin = build_orders(test)

    daily_units = test.groupby(test[DATE_COL].dt.normalize())["Order Item Quantity"].sum()
    capacity = args.capacity or int(math.ceil(np.percentile(daily_units, 75)))
    print(f"simulating {len(orders):,} orders over {(test[DATE_COL].max() - test[DATE_COL].min()).days} days; "
          f"capacity={capacity} units/day (p75 of daily volume)")

    results = {}
    for rule in RULES:
        result = simulate(orders, capacity, rule)
        results[rule] = result
        print(f"  {rule:<4s} on-time {result['on_time_rate']:6.1%}  "
              f"avg tardiness {result['avg_tardiness']:5.2f}d  "
              f"max {result['max_tardiness']:3d}d  WIP {result['avg_wip_orders']:6.0f}  "
              f"util {result['utilization']:.0%}")

    summary = pd.DataFrame(
        [
            {key: value for key, value in result.items() if key != "daily"}
            for result in results.values()
        ]
    )
    summary.to_csv(RESULTS / "ie_scheduling.csv", index=False)

    daily_all = pd.concat(
        [result["daily"].assign(rule=name) for name, result in results.items()], ignore_index=True
    )
    daily_all.to_csv(RESULTS / "ie_schedule_daily.csv", index=False)

    staffing = staffing_curve(test, args.units_per_worker, args.target_utilization)
    staffing.to_csv(RESULTS / "ie_staffing.csv", index=False)

    payload = {
        "orders": int(len(orders)),
        "capacity_units_per_day": int(capacity),
        "test_fraction": args.test_frac,
        "results": summary.to_dict(orient="records"),
        "staffing": {
            "units_per_worker": args.units_per_worker,
            "target_utilization": args.target_utilization,
            "peak_workers": int(staffing["workers_needed"].max()),
            "median_workers": float(staffing["workers_needed"].median()),
        },
    }
    (RESULTS / "ie_scheduling.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    plot(results, staffing, IMAGES)
    print(f"wrote {IMAGES / 'ie_scheduling.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
