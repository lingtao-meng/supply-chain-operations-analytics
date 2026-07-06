-- ============================================================
-- 供应链运营分析 — 数据库Schema (SQLite)
-- 星型模型: 5 维度表 + 1 事实表
-- ============================================================

-- ── 日期维度 ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS dim_date (
    date_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    order_date      DATE NOT NULL,
    order_year      INTEGER,
    order_month     INTEGER,
    order_quarter   INTEGER,
    order_weekday   INTEGER,
    order_month_name TEXT,
    year_month      TEXT
);

-- ── 客户维度 ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS dim_customer (
    customer_id      INTEGER PRIMARY KEY,
    customer_segment TEXT,
    customer_city    TEXT,
    customer_state   TEXT,
    customer_country TEXT,
    market           TEXT,
    order_region     TEXT
);

-- ── 产品维度 ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS dim_product (
    product_card_id     INTEGER PRIMARY KEY,
    product_category_id INTEGER,
    product_name        TEXT,
    category_name       TEXT,
    department_name     TEXT,
    product_price       REAL
);

-- ── 运输维度 ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS dim_shipping (
    shipping_mode_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    shipping_mode           TEXT UNIQUE,
    delivery_status         TEXT,
    days_shipping_scheduled REAL
);

-- ── 订单事实表（行级=订单×商品项）───────────────
CREATE TABLE IF NOT EXISTS fact_orders (
    fact_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id                INTEGER NOT NULL,
    order_date              DATE,
    shipping_date           DATE,
    order_status            TEXT,
    order_type              TEXT,
    order_city              TEXT,
    order_state             TEXT,
    order_country           TEXT,
    -- 外键
    customer_id             INTEGER,
    product_card_id         INTEGER,
    shipping_mode_id        INTEGER,
    -- 度量
    order_item_qty          INTEGER,
    product_price           REAL,
    order_total_value       REAL,
    order_item_profit       REAL,
    benefit_per_order       REAL,
    sales_per_customer      REAL,
    -- 交付指标
    days_shipping_real      REAL,
    days_shipping_scheduled REAL,
    delay_days              REAL,
    is_delayed              INTEGER,
    on_time                 INTEGER,
    delay_category          TEXT,
    late_delivery_risk      INTEGER,
    -- 日期维度
    order_year              INTEGER,
    order_month             INTEGER,
    year_month              TEXT,
    -- 外键约束
    FOREIGN KEY (customer_id)  REFERENCES dim_customer(customer_id),
    FOREIGN KEY (product_card_id) REFERENCES dim_product(product_card_id),
    FOREIGN KEY (shipping_mode_id) REFERENCES dim_shipping(shipping_mode_id)
);

-- ── 索引 ──────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_fact_date       ON fact_orders(order_date);
CREATE INDEX IF NOT EXISTS idx_fact_customer   ON fact_orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_fact_product    ON fact_orders(product_card_id);
CREATE INDEX IF NOT EXISTS idx_fact_shipping   ON fact_orders(shipping_mode_id);
CREATE INDEX IF NOT EXISTS idx_fact_delay      ON fact_orders(is_delayed);
CREATE INDEX IF NOT EXISTS idx_fact_year_month ON fact_orders(year_month);
