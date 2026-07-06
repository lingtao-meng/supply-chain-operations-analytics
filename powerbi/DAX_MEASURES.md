# Power BI DAX 度量 & 仪表板设计

## 数据连接

- 源: SQLite `supply_chain.db`
- 表: `dim_date`, `dim_customer`, `dim_product`, `dim_shipping`, `fact_orders`
- 关系: 星型模型 (fact → 各 dim 通过外键 1:N)

---

## 核心 DAX 度量

```dax
// ── KPI 卡片 ──────────────────────────────

Total Orders    = COUNT(fact_orders[order_id])
Total Revenue   = SUM(fact_orders[order_total_value])
Total Profit    = SUM(fact_orders[order_item_profit])
Avg Order Value = AVERAGE(fact_orders[order_total_value])

On Time Pct =
    DIVIDE(
        CALCULATE(COUNT(fact_orders[order_id]), fact_orders[on_time] = 1),
        COUNT(fact_orders[order_id])
    )

Avg Delivery Days = AVERAGE(fact_orders[days_shipping_real])

Avg Delay When Late =
    CALCULATE(
        AVERAGE(fact_orders[delay_days]),
        fact_orders[is_delayed] = 1
    )

Revenue At Risk =
    CALCULATE(
        SUM(fact_orders[order_total_value]),
        fact_orders[is_delayed] = 1
    )

Delayed Orders = CALCULATE(COUNT(fact_orders[order_id]), fact_orders[is_delayed] = 1)

On Time Orders = CALCULATE(COUNT(fact_orders[order_id]), fact_orders[on_time] = 1)

// ── 环比/同比 ─────────────────────────────

MoM On Time Change =
    VAR CurrentMonth = [On Time Pct]
    VAR PrevMonth =
        CALCULATE([On Time Pct],
            PREVIOUSMONTH(dim_date[order_date])
        )
    RETURN CurrentMonth - PrevMonth

YoY Revenue Change =
    VAR CurrentRevenue = [Total Revenue]
    VAR PrevYearRevenue =
        CALCULATE([Total Revenue],
            SAMEPERIODLASTYEAR(dim_date[order_date])
        )
    RETURN
        DIVIDE(CurrentRevenue - PrevYearRevenue, PrevYearRevenue)

// ── 条件格式 ──────────────────────────────

Delay Risk Color =
    SWITCH(TRUE(),
        [On Time Pct] >= 0.90, "Green",
        [On Time Pct] >= 0.80, "Yellow",
        "Red"
    )
```

---

## 仪表板 4 页设计

### 第 1 页 — 交付绩效 Overview

| 视觉 | 类型 | 维度 |
|------|------|------|
| 总订单 / 准时率 / 平均交货天数 / 收入风险 | KPI 卡片 | — |
| 月度准时率趋势 + 3月滚动均线 | 折线图 | year_month |
| 各运输方式准时率排名 | 横向条形图 | shipping_mode |
| 延迟等级分布 | 环形图 | delay_category |
| 各市场区域准时率热力 | 矩阵 | market × year_month |

### 第 2 页 — 运输与物流分析

| 视觉 | 类型 | 维度 |
|------|------|------|
| 运输方式 × 准时率 × 平均成本 | 散点图 | shipping_mode |
| 各区域准时率差距 | 条形图 | market |
| 平均交货天数 vs 计划天数对比 | 折线图 | shipping_mode |
| 运输方式订单量占比 | 饼图 | shipping_mode |
| 延迟订单按运输方式+区域下钻 | 矩阵 | shipping_mode × market |

### 第 3 页 — 品类 & 市场洞察

| 视觉 | 类型 | 维度 |
|------|------|------|
| 品类准时率排名 | 条形图 | category_name |
| 市场 × 品类交叉准时率 | 热力图 | market × category_name |
| 月度品类订单量趋势 | 面积图 | category_name, year_month |
| 全球市场收入分布 | 地图 | order_country |
| 各品类收入贡献 | 树状图 | category_name |

### 第 4 页 — 延迟根因分析

| 视觉 | 类型 | 维度 |
|------|------|------|
| 延迟贡献度瀑布图 | 瀑布图 | shipping_mode + delay_category |
| Top 15 延迟品类-市场组合 | 表格 | category_name × market |
| 延迟天数分布直方图 | 直方图 | delay_days |
| 准时率分位数视图 | 散点图 | category_name × market (by NTILE) |
| 各季度延迟率变化 | 折线图 | order_quarter |

---

## 交互功能

- **跨页联动**: 所有切片器（年份、运输方式、市场、品类）同步
- **下钻**: 从市场→国家→城市逐级下钻准时率
- **条件格式**: KPI 卡片根据阈值变色(绿≥90%, 黄≥80%, 红<80%)
- **提示工具**: 鼠标悬停显示订单数、平均延迟天数、收入影响
