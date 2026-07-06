# 供应链运营分析平台（SQL + Power BI）

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![SQLite](https://img.shields.io/badge/SQL-SQLite-003B57?logo=sqlite)](https://www.sqlite.org/)
[![Power BI](https://img.shields.io/badge/Power_BI-Dashboard-F2C811?logo=powerbi&logoColor=black)](https://powerbi.microsoft.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

基于 DataCo 全球供应链数据集（180K+ 订单）构建的端到端运营分析平台。从数据清洗到星型建模，从复杂 SQL 分析到 Power BI 可视化，完整覆盖供应链运营核心 KPI。

---

## 🎯 回答的核心业务问题

- 哪些运输方式的准时率最低？延迟集中在哪些市场和品类？
- 各市场区域的交付绩效差距有多大？趋势是在恶化还是改善？
- 延迟订单造成了多少收入风险？根因在运输、区域还是品类？
- 月度准时率是否有季节性波动？同比趋势如何？

---

## 📐 项目架构

```
DataCo CSV (180K, 53字段)
        │
        ▼
Python 数据清洗 (pandas)
  ├── 删除PII + 泄露特征
  ├── 缺失值处理
  ├── 衍生特征: 延迟天数/等级/准时标记/日期维度
  └── 输出: 清洗后CSV + 维度表CSV
        │
        ▼
SQLite 星型模型
  ├── dim_date       (日期维度)
  ├── dim_customer   (客户/市场维度)
  ├── dim_product    (产品/品类维度)
  ├── dim_shipping   (运输方式维度)
  └── fact_orders    (订单事实表, 180K行)
        │
        ▼
SQL 分析层 (10条复杂查询)
  ├── 窗口函数 (滚动均值/同比/环比)
  ├── CTE 多层聚合
  ├── NTILE 分位数排名
  └── 条件聚合 & Pareto分析
        │
        ▼
Power BI 仪表板 (4页)
  ├── 交付绩效 Overview
  ├── 运输与物流成本
  ├── 品类 & 市场洞察
  └── 延迟根因分析
```

---

## 📁 项目结构

```
supply-chain-operations-analytics/
├── data/
│   └── supply_chain.db          # SQLite数据库 (运行后生成)
├── python/
│   ├── 01_data_cleaning.py      # 数据清洗 + 特征工程
│   └── 02_load_to_db.py         # 星型模型入库
├── sql/
│   ├── 01_create_schema.sql     # 5维1事实表DDL + 索引
│   └── 02_analytics_queries.sql # 10条核心分析查询
├── powerbi/
│   └── DAX_MEASURES.md          # Power BI度量 + 仪表板设计
├── requirements.txt
└── README.md
```

---

## 🚀 快速开始

```bash
# 1. 克隆
git clone https://github.com/lingtao-meng/supply-chain-operations-analytics.git
cd supply-chain-operations-analytics

# 2. 安装依赖 (仅 pandas)
pip install -r requirements.txt

# 3. 数据清洗 (需先下载DataCo数据集到指定路径，或修改 INPUT_PATH)
python python/01_data_cleaning.py

# 4. 构建数据库
python python/02_load_to_db.py

# 5. 运行分析查询
sqlite3 data/supply_chain.db < sql/02_analytics_queries.sql

# 6. Power BI → 连接 data/supply_chain.db → 导入 DAX 度量
```

> **数据源**: [DataCo Supply Chain Dataset](https://www.kaggle.com/datasets/evilspirit05/datasupplychain) on Kaggle (93MB).  
> 如已有数据，修改 `python/01_data_cleaning.py` 中的 `INPUT_PATH` 变量。

---

## 📊 10条核心 SQL 分析

| # | 分析主题 | SQL 技术 |
|---|---------|---------|
| Q1 | 月度准时率 + 滚动3月均值 | `WINDOW FUNCTION (AVG OVER ROWS)`, `LAG` |
| Q2 | 运输方式准时率 & 排名 | `RANK`, `CASE WHEN`, `JOIN` |
| Q3 | 延迟根因: 运输 × 市场交叉 | `GROUP BY`, `HAVING` |
| Q4 | 品类月度需求 & 同比 | `LAG(..., 12)`, `PARTITION BY` |
| Q5 | 延迟天数分级统计 | `SUM OVER (PARTITION BY)`, `CASE WHEN` |
| Q6 | 高风险订单画像 | 多条件 `WHERE` + 多表 `JOIN` |
| Q7 | 品类-市场准时率分位数 | `NTILE(4)`, `RANK OVER (PARTITION BY)` |
| Q8 | 周度订单量 & 延迟趋势 | `strftime`, `GROUP BY` |
| Q9 | 产品延迟 Pareto 分析 | 累计占比 `SUM OVER (ORDER BY DESC)` |
| Q10 | 全局 KPI 汇总 | `COUNT DISTINCT`, 多维度聚合 |

---

## 📈 关键发现

| 洞察 | 数据 |
|------|------|
| Standard Class 是**准时率最高**的运输方式 | 60.2% 准时率，但平均4天 |
| First Class 准时率**为0%**，但仅2天平均交付 — 速度快但从不准时 | 0% on-time, n=9,307 |
| 超半数延迟集中在**1-3天**范围 | 53.7% 的延迟订单 |
| 拉美市场准时率**低于全球均值12个百分点** | market 维度可钻取 |
| 总体准时率约 **42.5%** | 180K 订单中有 103K 延迟 |

---

## 🛠 技术栈

| 层 | 技术 |
|---|------|
| 数据清洗 | Python (pandas, numpy) |
| 数据库 | SQLite3 (兼容 PostgreSQL 语法) |
| SQL 分析 | 窗口函数 / CTE / 子查询 / 聚合 / 排名 |
| 可视化 | Power BI (DAX + 交互仪表板) |

---

## 📄 License

MIT — 自由使用、修改和分享。

---

*Built as part of a supply chain analytics portfolio.*
