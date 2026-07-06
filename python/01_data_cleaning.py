"""
供应链运营分析 — 数据清洗模块
DataCo Global Supply Chain Dataset (180K+ orders)
"""
import pandas as pd
import numpy as np
import os

# ── 配置 ──────────────────────────────────────────
INPUT_PATH  = os.path.expanduser("~/Desktop/supply-chain-ml-pipeline/data/DataCoSupplyChainDataset.csv")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
ENCODING    = "latin1"

# 不需要的列（PII / 泄露特征 / 冗余）
DROP_COLS = [
    "Customer Email", "Customer Password", "Customer Fname", "Customer Lname",
    "Customer Street", "Customer Zipcode", "Product Description",
    "Order Item Discount", "Order Item Discount Rate",
    "Order Item Product Price", "Order Item Total",
    "Order Item Cardprod Id", "Order Item Id", "Product Image",
    "Order Zipcode", "Product Status",
    "Order Item Profit Ratio", "Order Profit Per Order",
    "Department Id", "Latitude", "Longitude",
]

# 列重命名 (Original → clean SQL name)
RENAME_MAP = {
    "Type":                            "order_type",
    "Days for shipping (real)":        "days_shipping_real",
    "Days for shipment (scheduled)":   "days_shipping_scheduled",
    "Benefit per order":               "benefit_per_order",
    "Sales per customer":              "sales_per_customer",
    "Delivery Status":                  "delivery_status",
    "Late_delivery_risk":              "late_delivery_risk",
    "Category Id":                      "category_id",
    "Category Name":                    "category_name",
    "Customer City":                    "customer_city",
    "Customer Country":                 "customer_country",
    "Customer Id":                      "customer_id",
    "Customer Segment":                 "customer_segment",
    "Customer State":                   "customer_state",
    "Market":                           "market",
    "Order City":                       "order_city",
    "Order Country":                    "order_country",
    "Order Customer Id":                "order_customer_id",
    "order date (DateOrders)":          "order_date",
    "Order Id":                         "order_id",
    "Order Item Quantity":              "order_item_qty",
    "Order Region":                     "order_region",
    "Order State":                      "order_state",
    "Order Status":                     "order_status",
    "Product Card Id":                  "product_card_id",
    "Product Category Id":              "product_category_id",
    "Product Name":                     "product_name",
    "Product Price":                    "product_price",
    "shipping date (DateOrders)":       "shipping_date",
    "Shipping Mode":                    "shipping_mode",
    "Department Name":                  "department_name",
    "Sales":                            "sales",
}

print("=" * 60)
print("供应链运营分析 — 数据清洗")
print("=" * 60)

# ── 1. 加载 ──────────────────────────────────────
print("\n[1/5] 加载原始数据 ...")
df = pd.read_csv(INPUT_PATH, encoding=ENCODING)
print(f"  行数: {len(df):,}  列数: {len(df.columns)}")

# ── 2. 删除无关列 ────────────────────────────────
print("\n[2/5] 删除无关列 ...")
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True, errors="ignore")
print(f"  剩余列: {len(df.columns)}")

# ── 3. 列重命名 ──────────────────────────────────
print("\n[3/5] 列重命名 ...")
df.rename(columns=RENAME_MAP, inplace=True)

# ── 4. 类型转换与清洗 ────────────────────────────
print("\n[4/5] 类型转换与清洗 ...")

# 日期列
for col in ["order_date", "shipping_date"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# 数值列
num_cols = ["days_shipping_real", "days_shipping_scheduled", "benefit_per_order",
            "sales_per_customer", "order_item_qty", "product_price",
            "late_delivery_risk", "sales"]
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 填充分类列缺失值
for col in ["customer_state", "customer_city", "order_state", "order_city"]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

# ── 5. 衍生特征（供应链运营指标）─────────────────
print("\n[5/5] 衍生特征 ...")

# 延迟天数（实际 vs 计划）
df["delay_days"] = df["days_shipping_real"] - df["days_shipping_scheduled"]
df["is_delayed"] = (df["delay_days"] > 0).astype(int)

# 准时交付
df["on_time"] = (df["delay_days"] <= 0).astype(int)

# 延迟等级
def delay_category(days):
    if pd.isna(days):
        return "Unknown"
    if days <= 0:
        return "On Time"
    elif days <= 3:
        return "1-3 Days"
    elif days <= 7:
        return "4-7 Days"
    else:
        return "8+ Days"

df["delay_category"] = df["delay_days"].apply(delay_category)

# 日期特征
if "order_date" in df.columns:
    df["order_year"]    = df["order_date"].dt.year
    df["order_month"]   = df["order_date"].dt.month
    df["order_quarter"] = df["order_date"].dt.quarter
    df["order_weekday"] = df["order_date"].dt.dayofweek
    df["order_month_name"] = df["order_date"].dt.month_name()
    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)

# 订单金额 = 单价 × 数量
df["order_total_value"] = df["product_price"] * df["order_item_qty"]

# ── 输出 ─────────────────────────────────────────
print("\n" + "=" * 60)
print("输出清洗后数据 ...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全量数据
clean_path = os.path.join(OUTPUT_DIR, "orders_cleaned.csv")
df.to_csv(clean_path, index=False, encoding="utf-8")
print(f"  ✓ orders_cleaned.csv  ({len(df):,} rows × {len(df.columns)} cols)")

# 维度表导出
dim_cols = {
    "dim_date":       ["order_date", "order_year", "order_month", "order_quarter",
                       "order_weekday", "order_month_name", "year_month"],
    "dim_customer":   ["customer_id", "customer_segment", "customer_city",
                       "customer_state", "customer_country", "market", "order_region"],
    "dim_product":    ["product_card_id", "product_category_id", "product_name",
                       "category_name", "department_name"],
    "dim_shipping":   ["shipping_mode", "delivery_status"],
}

for dim_name, cols in dim_cols.items():
    available = [c for c in cols if c in df.columns]
    dim_df = df[available].drop_duplicates()
    dim_path = os.path.join(OUTPUT_DIR, f"{dim_name}.csv")
    dim_df.to_csv(dim_path, index=False, encoding="utf-8")
    print(f"  ✓ {dim_name}.csv  ({len(dim_df)} rows)")

print("\n✅ 数据清洗完成！")
print(f"  → 下一步: python python/02_load_to_db.py")
