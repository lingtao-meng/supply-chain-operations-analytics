"""
供应链运营分析 — 交互式仪表板
Streamlit + Plotly | 4页看板
"""
import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(
    page_title="供应链运营分析平台",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 数据库连接 ────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "supply_chain.db")

@st.cache_resource
def load_data():
    if not os.path.exists(DB_PATH):
        st.error(f"数据库未找到: {DB_PATH}")
        st.info("请先运行: python python/01_data_cleaning.py && python python/02_load_to_db.py")
        st.stop()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT f.*, s.shipping_mode, s.delivery_status AS ship_status,
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

df = load_data()

# ── 侧边栏：全局筛选器 ────────────────────────────
st.sidebar.title("📦 供应链运营分析")
st.sidebar.markdown("---")

# 年份
years = sorted(df["order_year"].dropna().unique())
sel_years = st.sidebar.multiselect("年份", years, default=years)

# 运输方式
modes = sorted(df["shipping_mode"].dropna().unique())
sel_modes = st.sidebar.multiselect("运输方式", modes, default=modes)

# 市场
markets = sorted(df["market"].dropna().unique())
sel_markets = st.sidebar.multiselect("市场区域", markets, default=markets)

# 应用筛选
mask = (
    df["order_year"].isin(sel_years) &
    df["shipping_mode"].isin(sel_modes) &
    df["market"].isin(sel_markets)
)
dff = df[mask]

if len(dff) == 0:
    st.warning("当前筛选条件下无数据，请调整筛选器。")
    st.stop()

# ── 计算KPI ──────────────────────────────────────
total_orders   = dff["order_id"].nunique()      # 去重订单数
on_time_pct    = dff["on_time"].mean() * 100
avg_days       = dff["days_shipping_real"].mean()
total_revenue  = dff["order_total_value"].sum()
delayed_orders = dff["is_delayed"].sum()
revenue_risk   = dff[dff["is_delayed"] == 1]["order_total_value"].sum()

# ── 页面导航 ─────────────────────────────────────
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "导航",
    ["📊 交付绩效", "🚚 运输与物流", "🏷️ 品类与市场", "🔍 延迟根因"]
)

# ==================================================
# PAGE 1: 交付绩效
# ==================================================
if page == "📊 交付绩效":
    st.title("交付绩效 Overview")

    # KPI 卡片
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("总订单", f"{total_orders:,}")
    col2.metric("准时交付率", f"{on_time_pct:.1f}%")
    col3.metric("平均交付天数", f"{avg_days:.1f}天")
    col4.metric("总收入风险", f"${revenue_risk:,.0f}")
    col5.metric("延迟订单", f"{delayed_orders:,}")

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("月度准时交付率趋势")
        monthly = dff.groupby("year_month").agg(
            orders=("order_id", "count"),
            on_time_pct=("on_time", "mean")
        ).reset_index()
        monthly["on_time_pct"] = monthly["on_time_pct"] * 100
        monthly["rolling_3m"] = monthly["on_time_pct"].rolling(3).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["year_month"], y=monthly["on_time_pct"],
            mode="lines+markers", name="月度准时率",
            line=dict(color="#1f77b4", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=monthly["year_month"], y=monthly["rolling_3m"],
            mode="lines", name="3月滚动均值",
            line=dict(color="#ff7f0e", width=3, dash="dash")
        ))
        fig.update_layout(
            height=420,
            xaxis_title="", yaxis_title="准时率 (%)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("运输方式排名")
        mode_stats = dff.groupby("shipping_mode").agg(
            orders=("order_id", "count"),
            on_time_pct=("on_time", "mean")
        ).reset_index()
        mode_stats["on_time_pct"] = mode_stats["on_time_pct"] * 100
        mode_stats = mode_stats.sort_values("on_time_pct", ascending=True)

        fig = go.Figure(go.Bar(
            y=mode_stats["shipping_mode"],
            x=mode_stats["on_time_pct"],
            orientation="h",
            text=mode_stats["on_time_pct"].round(1).astype(str) + "%",
            textposition="outside",
            marker_color=["#d62728" if v < 30 else "#ff7f0e" if v < 50 else "#2ca02c" for v in mode_stats["on_time_pct"]]
        ))
        fig.update_layout(height=420, xaxis_title="准时率 (%)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # 底部: 延迟分布
    st.subheader("延迟等级分布")
    delay_dist = dff["delay_category"].value_counts()
    expected = ["On Time", "1-3 Days", "4-7 Days", "8+ Days"]
    delay_dist = delay_dist.reindex([c for c in expected if c in delay_dist.index])
    cols = st.columns(4)
    for i, (cat, count) in enumerate(delay_dist.items()):
        pct = count / total_orders * 100
        cols[i].metric(cat, f"{count:,}", f"{pct:.1f}%")

# ==================================================
# PAGE 2: 运输与物流
# ==================================================
elif page == "🚚 运输与物流":
    st.title("运输与物流成本分析")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("准时率 vs 平均交付天数")
        mode_stats = dff.groupby("shipping_mode").agg(
            orders=("order_id", "count"),
            on_time_pct=("on_time", "mean"),
            avg_days=("days_shipping_real", "mean"),
            total_rev=("order_total_value", "sum")
        ).reset_index()
        mode_stats["on_time_pct"] = mode_stats["on_time_pct"] * 100

        fig = px.scatter(
            mode_stats, x="avg_days", y="on_time_pct",
            size="orders", color="shipping_mode",
            text="shipping_mode",
            size_max=60,
            labels={"avg_days": "平均交付天数", "on_time_pct": "准时率 (%)"}
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("各市场区域准时率差距")
        market_stats = dff.groupby("market").agg(
            orders=("order_id", "count"),
            on_time_pct=("on_time", "mean")
        ).reset_index()
        market_stats["on_time_pct"] = market_stats["on_time_pct"] * 100
        global_avg = market_stats["on_time_pct"].mean()
        market_stats["vs_global"] = market_stats["on_time_pct"] - global_avg
        market_stats = market_stats.sort_values("vs_global")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=market_stats["market"],
            x=market_stats["vs_global"],
            orientation="h",
            text=market_stats["vs_global"].round(1).astype(str) + "%",
            marker_color=["#d62728" if v < 0 else "#2ca02c" for v in market_stats["vs_global"]]
        ))
        fig.add_vline(x=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            height=450,
            xaxis_title="vs 全球均值 (%)",
            yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("运输方式 × 市场 延迟热力图")
    pivot = dff.pivot_table(
        values="on_time", index="shipping_mode", columns="market",
        aggfunc="mean"
    ) * 100
    pivot = pivot.round(1)

    fig = px.imshow(
        pivot, text_auto=".1f", aspect="auto",
        color_continuous_scale="RdYlGn", range_color=[0, 100]
    )
    fig.update_layout(height=400)
    fig.update_coloraxes(colorbar_title="准时率 %")
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# PAGE 3: 品类与市场
# ==================================================
elif page == "🏷️ 品类与市场":
    st.title("品类与市场洞察")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("品类准时率排名")
        cat_stats = dff.groupby("category_name").agg(
            orders=("order_id", "count"),
            on_time_pct=("on_time", "mean"),
            total_rev=("order_total_value", "sum")
        ).reset_index()
        cat_stats["on_time_pct"] = cat_stats["on_time_pct"] * 100
        cat_stats = cat_stats[cat_stats["orders"] > 100].sort_values("on_time_pct", ascending=True).head(20)

        fig = go.Figure(go.Bar(
            y=cat_stats["category_name"],
            x=cat_stats["on_time_pct"],
            orientation="h",
            text=cat_stats["on_time_pct"].round(1).astype(str) + "%",
            textposition="outside"
        ))
        fig.update_layout(height=550, xaxis_title="准时率 (%)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("品类收入贡献 (树状图)")
        rev_data = dff.groupby("category_name")["order_total_value"].sum().reset_index()
        rev_data = rev_data.sort_values("order_total_value", ascending=False).head(15)

        fig = px.treemap(
            rev_data, path=["category_name"], values="order_total_value",
            color="order_total_value", color_continuous_scale="Blues"
        )
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("市场 × 品类 准时率热力图")
    mp = dff.pivot_table(
        values="on_time", index="market", columns="category_name",
        aggfunc="mean"
    )
    # top 15 categories by volume
    top_cats = dff["category_name"].value_counts().head(15).index
    mp = mp[top_cats] * 100

    fig = px.imshow(
        mp, text_auto=".1f", aspect="auto",
        color_continuous_scale="RdYlGn", range_color=[0, 100]
    )
    fig.update_layout(height=450)
    fig.update_coloraxes(colorbar_title="准时率 %")
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# PAGE 4: 延迟根因
# ==================================================
elif page == "🔍 延迟根因":
    st.title("延迟根因分析")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("延迟分布直方图")
        delayed = dff[dff["is_delayed"] == 1]
        fig = px.histogram(
            delayed, x="delay_days", nbins=30,
            labels={"delay_days": "延迟天数"},
            color_discrete_sequence=["#d62728"]
        )
        fig.add_vline(x=delayed["delay_days"].median(), line_dash="dash", line_color="black",
                      annotation_text=f"中位数: {delayed['delay_days'].median():.0f}天")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("延迟贡献度 (Pareto)")
        delay_profile = dff.groupby(["shipping_mode", "market"]).agg(
            delayed=("is_delayed", "sum"),
            total=("order_id", "count"),
            revenue=("order_total_value", "sum")
        ).reset_index()
        delay_profile["delay_pct"] = delay_profile["delayed"] / delay_profile["total"] * 100
        delay_profile = delay_profile.sort_values("delayed", ascending=False).head(15)
        delay_profile["cum_pct"] = delay_profile["delayed"].cumsum() / delay_profile["delayed"].sum() * 100

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=delay_profile["shipping_mode"] + " × " + delay_profile["market"],
            y=delay_profile["delayed"], name="延迟订单数",
            marker_color="#d62728"
        ))
        fig.add_trace(go.Scatter(
            x=delay_profile["shipping_mode"] + " × " + delay_profile["market"],
            y=delay_profile["cum_pct"], name="累计占比%",
            mode="lines+markers", line=dict(color="black", width=2),
            yaxis="y2"
        ))
        fig.update_layout(
            height=400,
            xaxis=dict(tickangle=-45),
            yaxis=dict(title="延迟订单数"),
            yaxis2=dict(title="累计%", range=[0, 110]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Top 15 延迟品类-市场组合")
        combo = dff.groupby(["category_name", "market"]).agg(
            orders=("order_id", "count"),
            on_time_pct=("on_time", "mean"),
            delay_days=("delay_days", "mean")
        ).reset_index()
        combo["on_time_pct"] = combo["on_time_pct"] * 100
        combo = combo[combo["orders"] > 50].sort_values("on_time_pct").head(15)

        fig = go.Figure(go.Bar(
            y=combo["category_name"] + " | " + combo["market"],
            x=combo["on_time_pct"],
            orientation="h",
            text=combo["on_time_pct"].round(1).astype(str) + "%",
            textposition="outside",
            marker_color="#d62728"
        ))
        fig.update_layout(height=500, xaxis_title="准时率 (%)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("各季度延迟率趋势")
        qtr = dff.groupby(["order_year", "order_quarter"]).agg(
            on_time_pct=("on_time", "mean"),
            orders=("order_id", "count")
        ).reset_index()
        qtr["on_time_pct"] = qtr["on_time_pct"] * 100
        qtr["yq"] = qtr["order_year"].astype(str) + "Q" + qtr["order_quarter"].astype(str)

        fig = go.Figure()
        for year in sorted(qtr["order_year"].unique()):
            yd = qtr[qtr["order_year"] == year]
            fig.add_trace(go.Scatter(
                x=yd["order_quarter"], y=yd["on_time_pct"],
                mode="lines+markers", name=str(year),
                line=dict(width=2)
            ))
        fig.update_layout(
            height=500,
            xaxis=dict(title="季度", tickmode="linear", dtick=1),
            yaxis=dict(title="准时率 (%)"),
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Footer ────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption(f"数据: DataCo Global Supply Chain | {total_orders:,} 条订单")
st.sidebar.caption(f"准时率: {on_time_pct:.1f}% | 平均交付: {avg_days:.1f}天")
