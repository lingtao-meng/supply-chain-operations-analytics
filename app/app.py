"""
供应链运营分析平台 — Streamlit 交互仪表板
自动下载数据 → 清洗 → 建模 → 可视化
"""
import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys, hashlib

st.set_page_config(
    page_title="供应链运营分析平台",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════
# 0. 自动初始化：下载数据 → 清洗 → 建库
# ═══════════════════════════════════════════════════════

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
DB_PATH      = os.path.join(DATA_DIR, "supply_chain.db")
CLEAN_CSV    = os.path.join(DATA_DIR, "orders_cleaned.csv")

def get_csv_hash(filepath):
    """计算CSV文件MD5，用于缓存失效判断"""
    return hashlib.md5(open(filepath, 'rb').read(4096)).hexdigest()

@st.cache_resource(show_spinner="正在下载数据集...")
def download_dataco():
    """从Kaggle下载DataCo数据集"""
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        import kagglehub
        path = kagglehub.dataset_download("evilspirit05/datasupplychain")
        csv_path = os.path.join(path, "DataCoSupplyChainDataset.csv")
        return csv_path
    except Exception:
        # kagglehub失败 → 尝试本地路径
        alt_paths = [
            os.path.expanduser("~/Desktop/supply-chain-ml-pipeline/data/DataCoSupplyChainDataset.csv"),
            os.path.expanduser("~/.cache/kagglehub/datasets/evilspirit05/datasupplychain/versions/1/DataCoSupplyChainDataset.csv"),
        ]
        for p in alt_paths:
            if os.path.exists(p):
                return p
        st.error("未找到DataCo数据集。请确保已下载或安装了kagglehub。")
        st.stop()

@st.cache_resource(show_spinner="正在清洗数据...")
def clean_data(input_csv):
    """数据清洗 + 特征工程"""
    import pandas as pd

    DROP_COLS = [
        "Customer Email","Customer Password","Customer Fname","Customer Lname",
        "Customer Street","Customer Zipcode","Product Description",
        "Order Item Discount","Order Item Discount Rate",
        "Order Item Product Price","Order Item Total",
        "Order Item Cardprod Id","Order Item Id","Product Image",
        "Order Zipcode","Product Status",
        "Order Item Profit Ratio","Order Profit Per Order",
        "Department Id","Latitude","Longitude",
    ]
    RENAME_MAP = {
        "Type":"order_type","Days for shipping (real)":"days_shipping_real",
        "Days for shipment (scheduled)":"days_shipping_scheduled",
        "Benefit per order":"benefit_per_order","Sales per customer":"sales_per_customer",
        "Delivery Status":"delivery_status","Late_delivery_risk":"late_delivery_risk",
        "Category Id":"category_id","Category Name":"category_name",
        "Customer City":"customer_city","Customer Country":"customer_country",
        "Customer Id":"customer_id","Customer Segment":"customer_segment",
        "Customer State":"customer_state","Market":"market",
        "Order City":"order_city","Order Country":"order_country",
        "Order Customer Id":"order_customer_id",
        "order date (DateOrders)":"order_date","Order Id":"order_id",
        "Order Item Quantity":"order_item_qty","Order Region":"order_region",
        "Order State":"order_state","Order Status":"order_status",
        "Product Card Id":"product_card_id","Product Category Id":"product_category_id",
        "Product Name":"product_name","Product Price":"product_price",
        "shipping date (DateOrders)":"shipping_date","Shipping Mode":"shipping_mode",
        "Department Name":"department_name","Sales":"sales",
    }

    df = pd.read_csv(input_csv, encoding="latin1")
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True, errors="ignore")
    df.rename(columns=RENAME_MAP, inplace=True)

    # 类型转换
    df["order_date"]    = pd.to_datetime(df["order_date"], errors="coerce")
    df["shipping_date"] = pd.to_datetime(df["shipping_date"], errors="coerce")
    for col in ["days_shipping_real","days_shipping_scheduled","benefit_per_order",
                "sales_per_customer","order_item_qty","product_price","late_delivery_risk","sales"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["customer_state","customer_city","order_state","order_city"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # 衍生特征
    df["delay_days"]   = df["days_shipping_real"] - df["days_shipping_scheduled"]
    df["is_delayed"]   = (df["delay_days"] > 0).astype(int)
    df["on_time"]      = (df["delay_days"] <= 0).astype(int)
    df["delay_category"] = pd.cut(df["delay_days"],
        bins=[-np.inf, 0, 3, 7, np.inf],
        labels=["On Time", "1-3 Days", "4-7 Days", "8+ Days"]).astype(str)
    df["order_year"]       = df["order_date"].dt.year
    df["order_month"]      = df["order_date"].dt.month
    df["order_quarter"]    = df["order_date"].dt.quarter
    df["order_weekday"]    = df["order_date"].dt.dayofweek
    df["year_month"]       = df["order_date"].dt.to_period("M").astype(str)
    df["order_total_value"] = df["product_price"] * df["order_item_qty"]

    return df

@st.cache_resource(show_spinner="正在构建星型模型数据库...")
def build_database(df):
    """建库 + 入库"""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)

    schema = os.path.join(PROJECT_ROOT, "sql", "01_create_schema.sql")
    if os.path.exists(schema):
        with open(schema) as f:
            conn.executescript(f.read())

    # 维度表
    conn.execute("DELETE FROM dim_date")
    df[["order_date","order_year","order_month","order_quarter",
        "order_weekday","year_month"]].drop_duplicates().to_sql(
        "dim_date", conn, if_exists="append", index=False)

    conn.execute("DELETE FROM dim_customer")
    df[["customer_id","customer_segment","customer_city",
        "customer_state","customer_country","market","order_region"]]\
        .drop_duplicates(subset="customer_id").to_sql(
        "dim_customer", conn, if_exists="append", index=False)

    conn.execute("DELETE FROM dim_product")
    df[["product_card_id","product_category_id","product_name",
        "category_name","department_name"]]\
        .drop_duplicates(subset="product_card_id").to_sql(
        "dim_product", conn, if_exists="append", index=False)

    conn.execute("DELETE FROM dim_shipping")
    for _, row in df[["shipping_mode","delivery_status"]]\
            .drop_duplicates(subset="shipping_mode").iterrows():
        conn.execute("INSERT INTO dim_shipping(shipping_mode,delivery_status) VALUES(?,?)",
                     (row["shipping_mode"], row["delivery_status"]))

    ship_map = dict(conn.execute(
        "SELECT shipping_mode, shipping_mode_id FROM dim_shipping").fetchall())

    fact_cols = ["order_id","order_date","shipping_date","order_status","order_type",
                 "order_city","order_state","order_country","customer_id","product_card_id",
                 "order_item_qty","product_price","order_total_value","benefit_per_order",
                 "sales_per_customer","days_shipping_real","days_shipping_scheduled",
                 "delay_days","is_delayed","on_time","delay_category",
                 "late_delivery_risk","order_year","order_month","year_month"]
    fact_df = df[[c for c in fact_cols if c in df.columns]].copy()
    if "shipping_mode" in df.columns:
        fact_df["shipping_mode_id"] = df["shipping_mode"].map(ship_map)

    conn.execute("DELETE FROM fact_orders")
    fact_df.to_sql("fact_orders", conn, if_exists="append", index=False)
    conn.close()
    return DB_PATH

@st.cache_resource
def load_from_db():
    """从数据库加载分析就绪数据"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT f.*, s.shipping_mode,
               c.customer_segment, c.customer_country, c.market, c.order_region,
               p.product_name, p.category_name, p.department_name
        FROM fact_orders f
        JOIN dim_shipping s  ON f.shipping_mode_id = s.shipping_mode_id
        JOIN dim_customer c  ON f.customer_id = c.customer_id
        JOIN dim_product p   ON f.product_card_id = p.product_card_id
        WHERE f.order_status = 'COMPLETE'
    """, conn)
    conn.close()
    return df


# ── 初始化流程 ─────────────────────────────────────
status = st.empty()
with st.spinner("正在初始化数据管道..."):
    raw_csv = download_dataco()
    df_clean = clean_data(raw_csv)
    db_path  = build_database(df_clean)
    df = load_from_db()
status.empty()

# ═══════════════════════════════════════════════════════
# 1. 侧边栏 + 全局筛选器
# ═══════════════════════════════════════════════════════

with st.sidebar:
    st.title("📦 供应链运营分析")
    st.caption(f"DataCo Global Supply Chain | {df['order_id'].nunique():,} 订单 | {len(df):,} 商品行")

    st.markdown("---")
    st.subheader("🔍 全局筛选")

    years  = sorted(df["order_year"].dropna().unique())
    modes  = sorted(df["shipping_mode"].dropna().unique())
    markets = sorted(df["market"].dropna().unique())

    sel_years   = st.multiselect("年份", years, default=years)
    sel_modes   = st.multiselect("运输方式", modes, default=modes)
    sel_markets = st.multiselect("市场区域", markets, default=markets)

    # 应用筛选
    mask = (
        df["order_year"].isin(sel_years) &
        df["shipping_mode"].isin(sel_modes) &
        df["market"].isin(sel_markets)
    )
    dff = df[mask]

    if len(dff) == 0:
        st.warning("无匹配数据")
        st.stop()

    # 实时KPI
    st.markdown("---")
    st.subheader("📈 实时指标")
    st.metric("筛选后订单", f"{dff['order_id'].nunique():,}")
    st.metric("准时率", f"{dff['on_time'].mean()*100:.1f}%")
    st.metric("平均交付", f"{dff['days_shipping_real'].mean():.1f}天")
    st.metric("收入风险", f"${dff.loc[dff['is_delayed']==1,'order_total_value'].sum():,.0f}")

    st.markdown("---")
    page = st.radio("📋 导航", ["📊 交付绩效", "🚚 运输与物流",
                                 "🏷️ 品类与市场", "🔍 延迟根因", "📐 数据概览"])

# ── KPI ───────────────────────────────────────────
total_orders   = dff["order_id"].nunique()
on_time_pct    = dff["on_time"].mean() * 100
avg_days       = dff["days_shipping_real"].mean()
delayed_orders = dff["is_delayed"].sum()
revenue_risk   = dff.loc[dff["is_delayed"] == 1, "order_total_value"].sum()

# ═══════════════════════════════════════════════════════
# PAGE 1: 交付绩效
# ═══════════════════════════════════════════════════════
if page == "📊 交付绩效":
    st.title("交付绩效 Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("总订单", f"{total_orders:,}")
    c2.metric("准时交付率", f"{on_time_pct:.1f}%",
              delta=f"{on_time_pct - 50:.1f}%" if on_time_pct > 50 else None)
    c3.metric("平均交付天数", f"{avg_days:.1f}天")
    c4.metric("收入风险", f"${revenue_risk:,.0f}")
    c5.metric("延迟订单行", f"{delayed_orders:,}")

    st.markdown("---")

    left, right = st.columns([2, 1])

    with left:
        st.subheader("月度准时率趋势")
        monthly = dff.groupby("year_month").agg(
            orders=("order_id","nunique"),
            on_time_pct=("on_time","mean")
        ).reset_index()
        monthly["on_time_pct"] *= 100
        monthly["rolling_3m"] = monthly["on_time_pct"].rolling(3, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly["year_month"], y=monthly["on_time_pct"],
            mode="lines+markers", name="月度准时率",
            line=dict(color="#1f77b4", width=2), marker=dict(size=4)))
        fig.add_trace(go.Scatter(x=monthly["year_month"], y=monthly["rolling_3m"],
            mode="lines", name="3月滚动均值",
            line=dict(color="#ff7f0e", width=3, dash="dash")))
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5)
        fig.update_layout(height=420, hovermode="x unified",
            legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("运输方式排名")
        mode_s = dff.groupby("shipping_mode").agg(
            orders=("order_id","nunique"), ot=("on_time","mean")).reset_index()
        mode_s["ot"] *= 100
        mode_s = mode_s.sort_values("ot", ascending=True)
        colors = ["#d62728" if v < 30 else "#ff7f0e" if v < 50 else "#2ca02c" for v in mode_s["ot"]]
        fig = go.Figure(go.Bar(y=mode_s["shipping_mode"], x=mode_s["ot"],
            orientation="h", text=mode_s["ot"].round(1).astype(str)+"%",
            textposition="outside", marker_color=colors))
        fig.update_layout(height=420, xaxis_title="准时率 (%)")
        st.plotly_chart(fig, use_container_width=True)

    # 延迟等级分布
    st.subheader("延迟等级分布")
    dist = dff["delay_category"].value_counts()
    cols = st.columns(4)
    for i, cat in enumerate(["On Time", "1-3 Days", "4-7 Days", "8+ Days"]):
        cnt = dist.get(cat, 0)
        cols[i].metric(cat, f"{cnt:,}", f"{cnt/len(dff)*100:.1f}%")

# ═══════════════════════════════════════════════════════
# PAGE 2: 运输与物流
# ═══════════════════════════════════════════════════════
elif page == "🚚 运输与物流":
    st.title("运输与物流成本分析")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("准时率 vs 平均交付天数")
        ms = dff.groupby("shipping_mode").agg(
            orders=("order_id","nunique"), ot=("on_time","mean"),
            avg_days=("days_shipping_real","mean"), rev=("order_total_value","sum")).reset_index()
        ms["ot"] *= 100
        fig = px.scatter(ms, x="avg_days", y="ot", size="orders",
            color="shipping_mode", text="shipping_mode", size_max=55,
            labels={"avg_days":"平均交付天数","ot":"准时率 (%)"})
        fig.update_traces(textposition="top center")
        fig.update_layout(height=430, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("各市场区域 vs 全球均值")
        mk = dff.groupby("market").agg(
            orders=("order_id","nunique"), ot=("on_time","mean")).reset_index()
        mk["ot"] *= 100
        gavg = mk["ot"].mean()
        mk["gap"] = mk["ot"] - gavg
        mk = mk.sort_values("gap")
        colors = ["#d62728" if v < 0 else "#2ca02c" for v in mk["gap"]]
        fig = go.Figure(go.Bar(y=mk["market"], x=mk["gap"], orientation="h",
            text=mk["gap"].round(1).astype(str)+"%", marker_color=colors))
        fig.add_vline(x=0, line_dash="dot", line_color="gray")
        fig.update_layout(height=430, xaxis_title=f"vs 全球均值 ({gavg:.1f}%)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("运输方式 × 市场 准时率热力图")
    pv = dff.pivot_table(values="on_time", index="shipping_mode", columns="market",
                          aggfunc="mean") * 100
    fig = px.imshow(pv.round(1), text_auto=".1f", aspect="auto",
                    color_continuous_scale="RdYlGn", range_color=[0, 100])
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════
# PAGE 3: 品类与市场
# ═══════════════════════════════════════════════════════
elif page == "🏷️ 品类与市场":
    st.title("品类与市场洞察")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("品类准时率排名")
        cs = dff.groupby("category_name").agg(
            orders=("order_id","nunique"), ot=("on_time","mean")).reset_index()
        cs["ot"] *= 100
        cs = cs[cs["orders"] > 50].sort_values("ot", ascending=True).head(20)
        fig = go.Figure(go.Bar(y=cs["category_name"], x=cs["ot"], orientation="h",
            text=cs["ot"].round(1).astype(str)+"%", textposition="outside"))
        fig.update_layout(height=550, xaxis_title="准时率 (%)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("品类收入贡献")
        rv = dff.groupby("category_name")["order_total_value"].sum().reset_index()
        rv = rv.sort_values("order_total_value", ascending=False).head(15)
        fig = px.treemap(rv, path=["category_name"], values="order_total_value",
                         color="order_total_value", color_continuous_scale="Blues")
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("市场 × 品类 准时率热力图")
    mp = dff.pivot_table(values="on_time", index="market", columns="category_name",
                          aggfunc="mean")
    top15 = dff["category_name"].value_counts().head(15).index
    mp = mp[[c for c in top15 if c in mp.columns]] * 100
    fig = px.imshow(mp.round(1), text_auto=".1f", aspect="auto",
                    color_continuous_scale="RdYlGn", range_color=[0, 100])
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════
# PAGE 4: 延迟根因
# ═══════════════════════════════════════════════════════
elif page == "🔍 延迟根因":
    st.title("延迟根因分析")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("延迟天数分布")
        delayed = dff[dff["is_delayed"] == 1]
        fig = px.histogram(delayed, x="delay_days", nbins=40,
            labels={"delay_days":"延迟天数"}, color_discrete_sequence=["#d62728"])
        fig.add_vline(x=delayed["delay_days"].median(), line_dash="dash",
            line_color="black", annotation_text=f"中位数: {delayed['delay_days'].median():.0f}天")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("延迟贡献度 Pareto")
        dp = dff.groupby(["shipping_mode","market"]).agg(
            delayed=("is_delayed","sum"), total=("order_id","count")).reset_index()
        dp = dp.sort_values("delayed", ascending=False).head(15)
        dp["cum"] = dp["delayed"].cumsum() / dp["delayed"].sum() * 100

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=dp["shipping_mode"]+" × "+dp["market"],
            y=dp["delayed"], name="延迟数", marker_color="#d62728"))
        fig.add_trace(go.Scatter(
            x=dp["shipping_mode"]+" × "+dp["market"],
            y=dp["cum"], name="累计%", mode="lines+markers",
            line=dict(color="black", width=2), yaxis="y2"))
        fig.update_layout(height=400, xaxis=dict(tickangle=-45),
            yaxis2=dict(range=[0,110]), legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Top 15 延迟品类-市场组合")
        cm = dff.groupby(["category_name","market"]).agg(
            orders=("order_id","nunique"), ot=("on_time","mean")).reset_index()
        cm["ot"] *= 100
        cm = cm[cm["orders"] > 30].sort_values("ot").head(15)
        fig = go.Figure(go.Bar(
            y=cm["category_name"]+" | "+cm["market"], x=cm["ot"],
            orientation="h", text=cm["ot"].round(1).astype(str)+"%",
            textposition="outside", marker_color="#d62728"))
        fig.update_layout(height=480, xaxis_title="准时率 (%)")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("各季度延迟率趋势")
        qtr = dff.groupby(["order_year","order_quarter"]).agg(
            ot=("on_time","mean"), orders=("order_id","nunique")).reset_index()
        qtr["ot"] *= 100

        fig = go.Figure()
        for yr in sorted(qtr["order_year"].unique()):
            yd = qtr[qtr["order_year"] == yr]
            fig.add_trace(go.Scatter(x=yd["order_quarter"], y=yd["ot"],
                mode="lines+markers", name=str(yr), line=dict(width=2)))
        fig.update_layout(height=480, xaxis=dict(title="季度", tickmode="linear", dtick=1),
            yaxis_title="准时率 (%)", legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════
# PAGE 5: 数据概览
# ═══════════════════════════════════════════════════════
elif page == "📐 数据概览":
    st.title("数据概览 & 质量报告")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("原始数据行", f"{len(df_clean):,}")
    c2.metric("清洗后列数", f"{len(df_clean.columns)}")
    c3.metric("唯一订单", f"{df_clean['order_id'].nunique():,}")
    c4.metric("数据时间范围", f"{df_clean['order_date'].dt.year.min():.0f}-{df_clean['order_date'].dt.year.max():.0f}")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("数据完整性")
        nulls = df_clean.isnull().sum()
        nulls = nulls[nulls > 0].sort_values(ascending=False)
        if len(nulls) > 0:
            st.dataframe(pd.DataFrame({"列": nulls.index, "缺失数": nulls.values, "缺失率%": (nulls/len(df_clean)*100).round(2)}),
                         use_container_width=True, hide_index=True)
        else:
            st.success("✅ 所有列无缺失值")

    with c2:
        st.subheader("数值列分布")
        num_col = st.selectbox("选择列", ["days_shipping_real","days_shipping_scheduled",
            "delay_days","order_total_value","benefit_per_order"])
        fig = px.histogram(df_clean, x=num_col, nbins=50, marginal="box")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("数据库 Schema")
    st.code("""
    ┌─────────────┐   ┌──────────────┐   ┌─────────────┐
    │  dim_date   │   │ dim_customer │   │ dim_product │
    │  date_id PK │   │ customer_id  │   │ product_id  │
    │  order_date │   │ segment      │   │ category    │
    │  year/month │   │ city/state   │   │ department  │
    │  quarter    │   │ country      │   │ price       │
    └──────┬──────┘   │ market/region│   └──────┬──────┘
           │          └──────┬───────┘          │
           │                 │                  │
           ▼                 ▼                  ▼
    ┌────────────────────────────────────────────────┐
    │              fact_orders (180K)                 │
    │  order_id | date | customer_id | product_id   │
    │  shipping_mode_id | qty | price | delay_days  │
    │  is_delayed | on_time | delay_category        │
    └──────────────────────┬─────────────────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │ dim_shipping   │
                  │ shipping_mode  │
                  └────────────────┘
    """, language=None)

    st.caption("星型模型: 5维度表 + 1事实表 | 6个索引 | SQLite")

# ═══════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("---")
    st.caption(f"仪表板就绪 | {len(df_clean):,}条数据 | SQLite + Streamlit + Plotly")
