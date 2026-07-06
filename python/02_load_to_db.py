"""
供应链运营分析 — 数据入库
将清洗后CSV加载到SQLite星型模型
"""
import sqlite3
import pandas as pd
import os

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH   = os.path.join(DATA_DIR, "supply_chain.db")
SCHEMA    = os.path.join(os.path.dirname(__file__), "..", "sql", "01_create_schema.sql")

print("=" * 60)
print("供应链运营分析 — 数据入库 (SQLite)")
print("=" * 60)

# ── 1. 建库 & Schema ─────────────────────────────
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)  # 重建，确保Schema干净
conn = sqlite3.connect(DB_PATH)
with open(SCHEMA, "r", encoding="utf-8") as f:
    conn.executescript(f.read())
print(f"✓ Schema 已创建 → {DB_PATH}")

# ── 2. 加载清洗后数据 ────────────────────────────
orders_path = os.path.join(DATA_DIR, "orders_cleaned.csv")
if not os.path.exists(orders_path):
    print(f"⚠️  未找到 {orders_path}，请先运行 01_data_cleaning.py")
    exit(1)

df = pd.read_csv(orders_path)
df["order_date"]    = pd.to_datetime(df["order_date"], errors="coerce")
df["shipping_date"] = pd.to_datetime(df["shipping_date"], errors="coerce")
print(f"✓ 加载: {len(df):,} rows")

# ── 3. 填充维度表 (INSERT保持Schema约束) ────────
print("\n[维度表]")

# dim_date
conn.execute("DELETE FROM dim_date")
date_df = df[["order_date", "order_year", "order_month", "order_quarter",
              "order_weekday", "order_month_name", "year_month"]].drop_duplicates()
date_df.to_sql("dim_date", conn, if_exists="append", index=False)
print(f"  dim_date:        {len(date_df):>6} rows")

# dim_customer
conn.execute("DELETE FROM dim_customer")
cus_df = df[["customer_id", "customer_segment", "customer_city",
             "customer_state", "customer_country", "market", "order_region"]].drop_duplicates(subset="customer_id")
cus_df.to_sql("dim_customer", conn, if_exists="append", index=False)
print(f"  dim_customer:     {len(cus_df):>6} rows")

# dim_product
conn.execute("DELETE FROM dim_product")
prod_df = df[["product_card_id", "product_category_id", "product_name",
              "category_name", "department_name"]].drop_duplicates(subset="product_card_id")
prod_df.to_sql("dim_product", conn, if_exists="append", index=False)
print(f"  dim_product:      {len(prod_df):>6} rows")

# dim_shipping (AUTOINCREMENT)
conn.execute("DELETE FROM dim_shipping")
for _, row in df[["shipping_mode", "delivery_status"]].drop_duplicates(subset="shipping_mode").iterrows():
    conn.execute(
        "INSERT INTO dim_shipping (shipping_mode, delivery_status) VALUES (?, ?)",
        (row["shipping_mode"], row["delivery_status"])
    )
n_ship = conn.execute("SELECT COUNT(*) FROM dim_shipping").fetchone()[0]
print(f"  dim_shipping:     {n_ship:>6} rows")

# ── 4. 填充事实表 ────────────────────────────────
print("\n[事实表]")

# 获取 shipping_mode_id 映射
ship_map = dict(conn.execute(
    "SELECT shipping_mode, shipping_mode_id FROM dim_shipping"
).fetchall())

fact_cols = [
    "order_id", "order_date", "shipping_date", "order_status", "order_type",
    "order_city", "order_state", "order_country",
    "customer_id", "product_card_id",
    "order_item_qty", "product_price", "order_total_value",
    "benefit_per_order", "sales_per_customer",
    "days_shipping_real", "days_shipping_scheduled",
    "delay_days", "is_delayed", "on_time", "delay_category",
    "late_delivery_risk", "order_year", "order_month", "year_month"
]

fact_cols = [c for c in fact_cols if c in df.columns]
fact_df = df[fact_cols].copy()
if "shipping_mode" in df.columns:
    fact_df["shipping_mode_id"] = df["shipping_mode"].map(ship_map)
    fact_cols.append("shipping_mode_id")

conn.execute("DELETE FROM fact_orders")
fact_df.to_sql("fact_orders", conn, if_exists="append", index=False)
print(f"  fact_orders:      {len(fact_df):>6} rows")

# ── 5. 验证 & 统计 ──────────────────────────────
print("\n" + "=" * 60)
print("[验证]")
for t in ["dim_date", "dim_customer", "dim_product", "dim_shipping", "fact_orders"]:
    n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t:20s} {n:>8,} rows")

delayed = conn.execute("SELECT COUNT(*) FROM fact_orders WHERE is_delayed=1").fetchone()[0]
total   = conn.execute("SELECT COUNT(*) FROM fact_orders").fetchone()[0]
print(f"\n  总订单:     {total:>8,}")
print(f"  延迟订单:   {delayed:>8,}")
print(f"  准时率:     {(1-delayed/total)*100:>7.1f}%")

conn.close()
print("\n✅ 数据入库完成！")
