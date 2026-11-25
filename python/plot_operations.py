"""Render the eight operations figures from completed order lines in the local warehouse."""
import argparse
from pathlib import Path
import sqlite3
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figure_style import format_figure  # noqa: E402

BLUE, DARK, GRAY = "#2a5570", "#25292b", "#8b959a"
QUERY = """
SELECT f.order_date, f.on_time, f.is_delayed, f.order_total_value,
       f.days_shipping_real, f.days_shipping_scheduled, s.shipping_mode,
       c.market, p.category_name, p.department_name
FROM fact_orders f
JOIN dim_shipping s USING (shipping_mode_id)
JOIN dim_customer c USING (customer_id)
JOIN dim_product p USING (product_card_id)
WHERE f.order_status='COMPLETE'
"""


def plot_gallery(data, output):
    output.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        for ax in fig.axes:
            if getattr(ax, "_colorbar", None) is None:
                ax.spines[["top", "right"]].set_visible(False)
        format_figure(fig)
        for suffix in ("", "_v2"):  # Keep both published gallery links current.
            fig.savefig(output / f"{name}{suffix}.png", dpi=260)
        plt.close(fig)

    def panel():
        return plt.subplots(figsize=(7.1, 3.6), layout="constrained")

    dates = pd.to_datetime(data.order_date)
    rate = data.on_time * 100
    monthly = rate.groupby(dates.dt.to_period("M")).mean()
    fig, ax = panel()
    time = monthly.index.to_timestamp()
    ax.plot(time, monthly, "-o", color=GRAY, lw=.75, ms=2.7, label="当月")
    ax.plot(time, monthly.rolling(3).mean(), color=BLUE, lw=1.6, label="近三个月均值")
    ax.axhline(rate.mean(), color=DARK, lw=.8, ls=":", label="全期总体")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set(ylabel="准时订单行占比（%）", xlabel="下单月份")
    ax.legend(ncol=3, fontsize=9)
    save(fig, "01_月度准时率趋势")

    rates = rate.groupby(data.shipping_mode).mean().sort_values()
    times = data.groupby("shipping_mode")[["days_shipping_scheduled", "days_shipping_real"]].mean().loc[rates.index]
    fig, (left, right) = plt.subplots(1, 2, sharey=True, figsize=(7.9, 3.25), layout="constrained")
    y = np.arange(len(rates))
    left.hlines(y, 0, rates, color="#b4c2cb", lw=1)
    left.plot(rates, y, "o", color=BLUE)
    left.set(yticks=y, yticklabels=rates.index, xlabel="准时订单行占比（%）", xlim=(-2,70))
    left.set_title("A    交付表现", loc="left", fontsize=11)
    right.hlines(y, times.iloc[:,0], times.iloc[:,1], color=GRAY)
    right.plot(times.iloc[:,0], y, "o", mec=DARK, mfc="white", label="承诺")
    right.plot(times.iloc[:,1], y, "D", color=BLUE, label="实际", ms=4)
    right.set(xlabel="平均运输天数（天）", ylim=(-.72,3.7))
    right.legend(ncol=2, fontsize=9, loc="lower right")
    right.set_title("B    承诺与实际", loc="left", fontsize=11)
    save(fig, "02_运输方式对比")

    gap = (rate.groupby(data.market).mean() - rate.mean()).sort_values()
    fig, ax = panel()
    ax.axvline(0, color=DARK, ls=":", lw=.85)
    ax.hlines(gap.index, 0, gap, color=GRAY)
    ax.plot(gap, gap.index, "o", color=BLUE)
    ax.set_xlabel(f"区域准时率与总体准时率（{rate.mean():.1f}%）之差（百分点）")
    save(fig, "03_市场区域差距")

    matrix = data.pivot_table(index="shipping_mode", columns="market", values="on_time") * 100
    fig, ax = panel()
    img = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set(xticks=range(len(matrix.columns)), xticklabels=matrix.columns,
           yticks=range(len(matrix)), yticklabels=matrix.index)
    for i, j in np.ndindex(matrix.shape):
        v = matrix.iloc[i,j]
        if pd.notna(v):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="white" if v>58 else DARK)
    fig.colorbar(img, ax=ax, label="准时订单行占比（%）")
    save(fig, "04_运输市场热力图")

    delayed = data.loc[data.is_delayed.eq(1)]
    counts = delayed.groupby(["shipping_mode", "market"]).size().nlargest(12)
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8.5, 4.4), layout="constrained")
    y = np.arange(len(counts))
    axes[0].barh(y, counts, color="#637e8d", height=.57)
    axes[0].set(yticks=y, yticklabels=[" · ".join(k) for k in counts.index], xlabel="延迟订单行（行）")
    axes[0].invert_yaxis()
    axes[1].plot(100 * counts.cumsum()/len(delayed), y, "o-", color=BLUE, lw=1, ms=3.5)
    axes[1].set(xlim=(0,100), xlabel="累计占全部延迟订单行（%）")
    save(fig, "05_延迟Pareto分析")

    factors = {"运输方式":data.shipping_mode, "部门":data.department_name,
               "星期":dates.dt.weekday, "季度":dates.dt.quarter, "市场":data.market}
    ranges = {label: group.max()-group.min() for label, key in factors.items()
              for group in [rate.groupby(key).mean()]}
    spread = pd.Series(ranges).sort_values()
    fig, ax = panel()
    ax.hlines(spread.index, 0, spread, color=GRAY)
    ax.plot(spread, spread.index, "o", color=BLUE)
    ax.set_xlabel("同一维度各组准时率的最大差值（百分点）")
    save(fig, "06_因素关联强度")

    revenue = data.groupby("category_name").order_total_value.sum().nlargest(12).sort_values()/1e6
    fig, ax = panel()
    ax.barh(revenue.index, revenue, color="#59778a", height=.58)
    ax.set_xlabel("已完成订单行金额合计（百万美元）")
    save(fig, "07_品类收入排名")

    by_quarter = rate.groupby([dates.dt.year.rename("year"), dates.dt.quarter.rename("quarter")]).mean()
    fig, ax = panel()
    for (year, series), marker, style in zip(by_quarter.groupby(level=0), ["o","s","^","D"], ["-","--","-.",":"]):
        ax.plot(series.index.get_level_values(1), series, marker=marker, ls=style,
                color=BLUE if year%2 else DARK, ms=4, lw=1, label=str(year))
    ax.set(xticks=[1,2,3,4], xlabel="下单季度", ylabel="准时订单行占比（%）")
    ax.legend(ncol=4, fontsize=9)
    save(fig, "08_季度趋势")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=ROOT/"data/supply_chain.db")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"images")
    args = parser.parse_args()
    with sqlite3.connect(args.database.resolve().as_uri()+"?mode=ro", uri=True) as connection:
        plot_gallery(pd.read_sql_query(QUERY, connection), args.output_dir)
