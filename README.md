# 供应链运营分析平台

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![SQLite](https://img.shields.io/badge/SQL-SQLite-003B57?logo=sqlite)](https://www.sqlite.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

用 DataCo 全球供应链数据集（180K+ 订单，53 个字段）做的运营分析项目。从数据清洗、星型建模、SQL 查询到 Streamlit 仪表板，整套流程都覆盖了。

---

## 做了什么

数据清洗后用 SQLite 搭了一个星型模型（5 张维度表 + 1 张事实表），写了 10 条分析查询，最后用 Streamlit + Plotly 做了一个 5 页的交互仪表板。

仪表板跑起来之后能看出来一些有意思的东西。比如 First Class 这个运输方式，SLA 承诺 1 天送达，但实际上平均要 2 天，准时率是 0%——不是说它慢，是它的承诺不现实。Second Class 也好不到哪去，承诺 2 天，实际平均 3.9 天。Standard Class 反而是最靠谱的，说 4 天就 4 天，准时率 60%。

超一半的延迟集中在 1-3 天，属于轻微延迟，不是大问题但量大，流程上优化一下应该见效很快。LATAM 市场的准时率比全球均值低一截，物流基础设施的问题比较明显。

仪表板上有自动生成的文字洞察，会根据你选的筛选条件变化。比如只选某个年份和某个市场，它会告诉你这个范围内哪个运输方式最好、延迟集中在哪里、大概多少收入在风险敞口里。

---

## 项目结构

```
├── app/app.py                       Streamlit 仪表板（5 页）
├── python/
│   ├── 01_data_cleaning.py          数据清洗 + 特征工程
│   └── 02_load_to_db.py            星型模型入库
├── sql/
│   ├── 01_create_schema.sql         建表 DDL + 索引
│   └── 02_analytics_queries.sql     10 条分析查询
├── notebooks/
│   └── supply_chain_deep_analysis.ipynb  深度分析 + ROI 估算
├── tests/
│   └── test_sql_queries.py          SQL 查询验证
├── images/                          仪表板截图
├── requirements.txt
└── setup.sh                         一键初始化
```

---

## 怎么跑

```bash
git clone https://github.com/lingtao-meng/supply-chain-operations-analytics.git
cd supply-chain-operations-analytics

pip install -r requirements.txt

# 先跑数据清洗（需要先下载 DataCo 数据集，改一下 INPUT_PATH）
python python/01_data_cleaning.py
python python/02_load_to_db.py

# 启动仪表板
streamlit run app/app.py

# 或者一步到位
bash setup.sh && streamlit run app/app.py
```

数据集从 Kaggle 下载：[DataCo Supply Chain Dataset](https://www.kaggle.com/datasets/evilspirit05/datasupplychain)（93MB，180,519 条记录）。

如果本地已经有这个数据集，App 启动时会自动找到它，不需要手动跑清洗脚本。

---

## SQL 分析

10 条查询放在 `sql/02_analytics_queries.sql`，主要用到的写法：

- 窗口函数：滚动均值、环比、同比（LAG 按品类分区回溯 12 个月）
- CTE 做多层聚合，再套 NTILE 分位数排名
- Pareto 分析看延迟集中在哪些运输×市场组合上
- 条件聚合做延迟分级统计

验证查询跑通：

```bash
python tests/test_sql_queries.py
```

---

## 仪表板页面

| 页面 | 内容 |
|------|------|
| 交付绩效 | KPI 卡片、月度趋势折线、运输方式排名、延迟等级分布 |
| 运输与物流 | 准时率 vs 交付天数散点、市场区域差距、运输×市场热力图 |
| 品类与市场 | 品类准时率排名、收入树状图、品类×市场热力图 |
| 延迟根因 | 延迟分布直方、Pareto 贡献图、因素关联强度、季度趋势 |
| 数据概览 | 缺失值报告、数值分布、Schema 图 |

侧边栏有年份、运输方式、市场三个筛选器，全局联动。页面顶部有自动生成的文字洞察。筛选后的数据可以一键导出 CSV。

---

## 截图

| | |
|:---:|:---:|
| ![月度趋势](images/01_月度准时率趋势.png) | ![运输对比](images/02_运输方式对比.png) |
| ![市场差距](images/03_市场区域差距.png) | ![热力图](images/04_运输市场热力图.png) |
| ![Pareto](images/05_延迟Pareto分析.png) | ![因素关联](images/06_因素关联强度.png) |

---

## 技术栈

Python (pandas, numpy) → SQLite → Streamlit + Plotly → Jupyter

---

MIT License
