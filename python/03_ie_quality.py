#!/usr/bin/env python3
"""IE quality engineering for the order-to-delivery process.

Three classical tools, applied to DataCo:

1. **Laney p′ control chart** — weekly/monthly late rate with binomial limits
   corrected for overdispersion.  With 5,000+ orders per week a plain p-chart
   flags almost everything; Laney's correction (divide by the observed
   standard deviation of the standardised points) is the right tool.
2. **Process capability** — transit days vs the promised days (USL), reporting
   yield, Ppk, sigma level and DPMO per shipping mode.
3. **Pareto** — where the late orders actually come from.

Western Electric rules (1 point beyond 3σ, 2 of 3 beyond 2σ, 8 on one side)
are evaluated on the Laney-standardised points.

Usage
-----
    python python/03_ie_quality.py
    python python/03_ie_quality.py --data /path/to/DataCoSupplyChainDataset.csv --freq M
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
IMAGES = ROOT / "images"

from ie_common import DATE_COL, SCHED_COL, SHIP_COL, find_data, load_orders  # noqa: E402


def western_electric(z: np.ndarray) -> dict:
    """Return rule flags for a standardised series (rule 1/2/3)."""
    z = np.asarray(z, dtype=float)
    n = len(z)
    rule1 = np.abs(z) > 3
    rule2 = np.zeros(n, dtype=bool)
    for i in range(2, n):
        window = z[i - 2 : i + 1]
        if np.sum(window > 2) >= 2 or np.sum(window < -2) >= 2:
            rule2[i] = True
    rule3 = np.zeros(n, dtype=bool)
    for i in range(7, n):
        window = z[i - 7 : i + 1]
        if np.all(window > 0) or np.all(window < 0):
            rule3[i] = True
    return {"rule1": rule1, "rule2": rule2, "rule3": rule3, "any": rule1 | rule2 | rule3}


def laney_p_chart(df: pd.DataFrame, freq: str = "M") -> tuple[pd.DataFrame, dict]:
    """p′ chart: binomial limits inflated by the observed overdispersion."""
    period = df[DATE_COL].dt.to_period(freq)
    grouped = (
        df.groupby(period)
        .agg(orders=("delay_days", "size"), late=("delay_days", lambda s: int((s > 0).sum())))
        .reset_index()
    )
    grouped["period"] = grouped[DATE_COL].astype(str)
    grouped["p"] = grouped["late"] / grouped["orders"]

    pbar = float(grouped["late"].sum() / grouped["orders"].sum())
    sigma = np.sqrt(pbar * (1 - pbar) / grouped["orders"])
    z = (grouped["p"] - pbar) / sigma
    sigma_z = float(np.std(z, ddof=1)) if len(z) > 1 else 1.0
    factor = max(1.0, sigma_z)  # Laney: never tighten below the binomial limits
    standard_out = int(((grouped["p"] > pbar + 3 * sigma) | (grouped["p"] < pbar - 3 * sigma)).sum())
    grouped["ucl"] = pbar + 3 * factor * sigma
    grouped["lcl"] = np.clip(pbar - 3 * factor * sigma, 0, None)
    grouped["z"] = (grouped["p"] - pbar) / (sigma * factor)

    flags = western_electric(grouped["z"].to_numpy())
    for key, values in flags.items():
        grouped[f"we_{key}"] = values
    grouped["out_of_control"] = (grouped["p"] > grouped["ucl"]) | (grouped["p"] < grouped["lcl"])

    summary = {
        "period_freq": freq,
        "periods": int(len(grouped)),
        "pbar": pbar,
        "laney_factor": factor,
        "standard_p_out_of_control": standard_out,
        "out_of_control": int(grouped["out_of_control"].sum()),
        "we_rule1": int(flags["rule1"].sum()),
        "we_rule2": int(flags["rule2"].sum()),
        "we_rule3": int(flags["rule3"].sum()),
    }
    return grouped, summary


def capability(df: pd.DataFrame) -> pd.DataFrame:
    """One-sided capability of transit days against the promised days (USL)."""
    rows = []
    for mode, group in df.groupby("Shipping Mode", sort=True):
        usl = float(group[SCHED_COL].iloc[0])
        actual = group["transit_days"].to_numpy(dtype=float)
        mean, std = float(actual.mean()), float(actual.std(ddof=1))
        yield_rate = float((actual <= usl).mean())
        ppk = (usl - mean) / (3 * std) if std > 0 else float("nan")
        z_value = float(stats.norm.ppf(np.clip(yield_rate, 1e-9, 1 - 1e-9)))
        rows.append(
            {
                "shipping_mode": mode,
                "promised_days": usl,
                "mean_transit_days": mean,
                "std_transit_days": std,
                "yield": yield_rate,
                "ppk": ppk,
                "sigma_level_1_5_shift": z_value + 1.5,
                "dpmo": (1 - yield_rate) * 1_000_000,
                "n": int(len(group)),
                "note": "zero variance" if std == 0 else "",
            }
        )
    return pd.DataFrame(rows).sort_values("yield", ascending=False)


def pareto(df: pd.DataFrame, by: str) -> pd.DataFrame:
    late = df[df["delay_days"] > 0]
    counts = late[by].value_counts()
    table = counts.rename_axis(by).reset_index(name="late_orders")
    table["share"] = table["late_orders"] / table["late_orders"].sum()
    table["cumulative_share"] = table["share"].cumsum()
    return table


def plot(chart: pd.DataFrame, cap: pd.DataFrame, par: pd.DataFrame, images: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    ax = axes[0]
    x = np.arange(len(chart))
    ax.plot(x, chart["p"], color="#1f6f78", marker="o", markersize=3, label="late rate")
    ax.plot(x, chart["ucl"], color="#9b3b3b", linestyle="--", linewidth=1, label="Laney UCL")
    ax.plot(x, chart["lcl"], color="#9b3b3b", linestyle="--", linewidth=1, label="Laney LCL")
    ax.axhline(chart["p"].mean(), color="#7c8a99", linestyle=":", linewidth=1, label="p̄")
    bad = chart[chart["out_of_control"]]
    ax.scatter(bad.index, bad["p"], color="#9b3b3b", zorder=5, label="out of control")
    step = max(1, len(chart) // 8)
    ax.set_xticks(x[::step], chart["period"][::step], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("late rate")
    ax.set_title("Laney p′ chart — weekly/monthly late rate")
    ax.legend(fontsize=7)

    ax2 = axes[1]
    view = cap.iloc[::-1]
    ax2.barh(view["shipping_mode"], view["yield"] * 100, color="#1f6f78")
    for i, (_, row) in enumerate(view.iterrows()):
        yield_rate = float(row["yield"])
        label = f"{yield_rate:.1%}"
        if np.isfinite(row["ppk"]):
            label += f"  Ppk={row['ppk']:.2f}"
        ax2.text(yield_rate * 100 + 1, i, label, va="center", fontsize=8)
    ax2.set_xlim(0, 115)
    ax2.set_xlabel("share of orders within the promised days (%)")
    ax2.set_title("Process capability: transit vs promise")

    ax3 = axes[2]
    ax3.bar(range(len(par)), par["share"] * 100, color="#2f5d8c")
    ax3.plot(range(len(par)), par["cumulative_share"] * 100, color="#d99a2b", marker="o")
    ax3.set_xticks(range(len(par)), par[par.columns[0]], rotation=30, ha="right", fontsize=8)
    ax3.set_ylabel("share of late orders (%)")
    ax3.set_title("Pareto of late orders by shipping mode")
    ax3.axhline(80, color="#c9ced6", linestyle=":", linewidth=1)

    fig.tight_layout()
    fig.savefig(images / "ie_quality.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None)
    ap.add_argument("--freq", default="M", choices=["W", "M", "Q"])
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)

    data_path = find_data(args.data)
    df = load_orders(data_path)
    chart, summary = laney_p_chart(df, args.freq)
    cap = capability(df)
    par = pareto(df, "Shipping Mode")

    chart.to_csv(RESULTS / "ie_p_chart.csv", index=False)
    cap.to_csv(RESULTS / "ie_capability.csv", index=False)
    par.to_csv(RESULTS / "ie_pareto.csv", index=False)
    payload = {
        "data": str(data_path),
        "p_chart": summary,
        "capability": cap.astype(object).where(pd.notna(cap), None).to_dict(orient="records"),
        "pareto": par.to_dict(orient="records"),
    }
    (RESULTS / "ie_quality.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )

    print(f"data: {data_path}")
    print(f"p-chart: p̄={summary['pbar']:.1%}, Laney factor={summary['laney_factor']:.2f}, "
          f"out-of-control={summary['out_of_control']}/{summary['periods']}")
    print(cap.to_string(index=False))
    print("\nPareto (shipping mode):")
    print(par.to_string(index=False))

    plot(chart, cap, par, IMAGES)
    print(f"wrote {IMAGES / 'ie_quality.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
