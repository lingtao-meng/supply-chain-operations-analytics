-- ============================================================
-- 供应链运营分析 — 核心分析查询 (10条)
-- 每条对应一个供应链KPI或业务洞察
-- ============================================================

-- ──────────────────────────────────────────────────────────
-- Q1: 月度准时交付率 & 滚动3月均值（窗口函数）
-- ──────────────────────────────────────────────────────────
WITH monthly_ot AS (
    SELECT
        order_year,
        order_month,
        COUNT(*)                                                AS total_orders,
        SUM(on_time)                                            AS on_time_orders,
        ROUND(SUM(on_time) * 100.0 / COUNT(*), 2)              AS on_time_pct
    FROM fact_orders
    WHERE order_status = 'COMPLETE'
    GROUP BY order_year, order_month
)
SELECT
    order_year,
    order_month,
    total_orders,
    on_time_pct,
    ROUND(AVG(on_time_pct) OVER (
        ORDER BY order_year, order_month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2) AS rolling_3m_on_time_pct,
    ROUND(on_time_pct - LAG(on_time_pct) OVER (
        ORDER BY order_year, order_month
    ), 2) AS mom_change_pct
FROM monthly_ot
ORDER BY order_year, order_month;


-- ──────────────────────────────────────────────────────────
-- Q2: 各运输方式准时率 & 平均延迟天数对比 (含排名)
-- ──────────────────────────────────────────────────────────
SELECT
    s.shipping_mode,
    COUNT(*)                                                            AS order_count,
    ROUND(SUM(f.on_time) * 100.0 / COUNT(*), 2)                        AS on_time_pct,
    ROUND(AVG(f.days_shipping_real), 1)                                 AS avg_shipping_days,
    ROUND(AVG(CASE WHEN f.is_delayed = 1 THEN f.delay_days END), 1)    AS avg_delay_when_late,
    ROUND(SUM(f.order_total_value), 0)                                  AS total_revenue,
    RANK()  OVER (ORDER BY SUM(f.on_time) * 100.0 / COUNT(*) DESC)     AS rank_on_time,
    RANK()  OVER (ORDER BY COUNT(*) DESC)                               AS rank_volume
FROM fact_orders f
JOIN dim_shipping s ON f.shipping_mode_id = s.shipping_mode_id
WHERE f.order_status = 'COMPLETE'
GROUP BY s.shipping_mode
ORDER BY on_time_pct DESC;


-- ──────────────────────────────────────────────────────────
-- Q3: 延迟根因拆解 — 运输方式 × 市场区域 交叉分析
-- ──────────────────────────────────────────────────────────
SELECT
    s.shipping_mode,
    c.market,
    COUNT(*)                                                            AS order_count,
    ROUND(SUM(f.on_time) * 100.0 / COUNT(*), 2)                        AS on_time_pct,
    ROUND(AVG(f.delay_days), 1)                                         AS avg_delay_days,
    ROUND(SUM(f.order_total_value), 0)                                  AS total_revenue
FROM fact_orders f
JOIN dim_shipping s ON f.shipping_mode_id = s.shipping_mode_id
JOIN dim_customer c ON f.customer_id = c.customer_id
WHERE f.order_status = 'COMPLETE'
GROUP BY s.shipping_mode, c.market
HAVING COUNT(*) > 100
ORDER BY on_time_pct ASC
LIMIT 20;


-- ──────────────────────────────────────────────────────────
-- Q4: 各品类月度需求波动 & 延迟率趋势 (同比)
-- ──────────────────────────────────────────────────────────
WITH monthly AS (
    SELECT
        p.category_name,
        f.order_year,
        f.order_month,
        COUNT(*)                                                    AS orders,
        ROUND(SUM(f.on_time) * 100.0 / COUNT(*), 2)                AS on_time_pct
    FROM fact_orders f
    JOIN dim_product p ON f.product_card_id = p.product_card_id
    WHERE f.order_status = 'COMPLETE'
    GROUP BY p.category_name, f.order_year, f.order_month
)
SELECT
    category_name,
    order_year,
    order_month,
    orders,
    on_time_pct,
    LAG(orders, 12) OVER (
        PARTITION BY category_name
        ORDER BY order_year, order_month
    ) AS orders_last_year_same_month,
    ROUND(
        (orders - LAG(orders, 12) OVER (
            PARTITION BY category_name ORDER BY order_year, order_month
        )) * 100.0 / NULLIF(LAG(orders, 12) OVER (
            PARTITION BY category_name ORDER BY order_year, order_month
        ), 0), 2
    ) AS yoy_change_pct
FROM monthly
ORDER BY category_name, order_year, order_month;


-- ──────────────────────────────────────────────────────────
-- Q5: 延迟天数分级统计 (条件聚合)
-- ──────────────────────────────────────────────────────────
SELECT
    delay_category,
    COUNT(*)                                                            AS order_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)                 AS pct_of_total,
    ROUND(AVG(days_shipping_real), 1)                                   AS avg_shipping_days,
    ROUND(AVG(order_total_value), 2)                                    AS avg_order_value,
    ROUND(SUM(order_total_value), 0)                                    AS total_revenue_impact,
    ROUND(AVG(benefit_per_order), 2)                                    AS avg_benefit
FROM fact_orders
WHERE order_status = 'COMPLETE'
GROUP BY delay_category
ORDER BY
    CASE delay_category
        WHEN 'On Time'       THEN 1
        WHEN '1-3 Days' THEN 2
        WHEN '4-7 Days' THEN 3
        WHEN '8+ Days'  THEN 4
        ELSE 5
    END;


-- ──────────────────────────────────────────────────────────
-- Q6: 延迟高风险订单画像 (多条件组合)
-- ──────────────────────────────────────────────────────────
SELECT
    s.shipping_mode,
    c.market,
    p.category_name,
    COUNT(*)                                                  AS delayed_orders,
    ROUND(AVG(f.delay_days), 1)                               AS avg_delay,
    ROUND(AVG(f.order_total_value), 2)                        AS avg_value,
    ROUND(SUM(f.order_item_profit), 0)                        AS total_profit_at_risk
FROM fact_orders f
JOIN dim_shipping s ON f.shipping_mode_id = s.shipping_mode_id
JOIN dim_customer c ON f.customer_id = c.customer_id
JOIN dim_product p ON f.product_card_id = p.product_card_id
WHERE f.is_delayed = 1
  AND f.order_status = 'COMPLETE'
GROUP BY s.shipping_mode, c.market, p.category_name
HAVING COUNT(*) > 50
ORDER BY delayed_orders DESC
LIMIT 15;


-- ──────────────────────────────────────────────────────────
-- Q7: 准时率分位数 — 品类在市场的排名
-- ──────────────────────────────────────────────────────────
WITH cat_market AS (
    SELECT
        p.category_name,
        c.market,
        COUNT(*)                                                AS orders,
        ROUND(SUM(f.on_time) * 100.0 / COUNT(*), 2)            AS on_time_pct
    FROM fact_orders f
    JOIN dim_product p  ON f.product_card_id = p.product_card_id
    JOIN dim_customer c ON f.customer_id = c.customer_id
    WHERE f.order_status = 'COMPLETE'
    GROUP BY p.category_name, c.market
    HAVING COUNT(*) > 200
)
SELECT
    category_name,
    market,
    orders,
    on_time_pct,
    NTILE(4) OVER (PARTITION BY market ORDER BY on_time_pct DESC) AS quartile,
    RANK()   OVER (PARTITION BY market ORDER BY on_time_pct DESC) AS rank_in_market
FROM cat_market
ORDER BY market, on_time_pct DESC;


-- ──────────────────────────────────────────────────────────
-- Q8: 周度订单量 & 延迟率趋势 (季节性)
-- ──────────────────────────────────────────────────────────
SELECT
    strftime('%W', order_date)  AS week_number,
    order_year,
    COUNT(*)                                                    AS orders,
    ROUND(SUM(on_time) * 100.0 / COUNT(*), 2)                  AS on_time_pct,
    SUM(order_total_value)                                      AS weekly_revenue,
    ROUND(AVG(order_total_value), 2)                            AS avg_order_value
FROM fact_orders
WHERE order_status = 'COMPLETE'
GROUP BY order_year, week_number
ORDER BY order_year, week_number;


-- ──────────────────────────────────────────────────────────
-- Q9: 供应商/产品延迟贡献度 (Pareto分析)
-- ──────────────────────────────────────────────────────────
WITH product_delay AS (
    SELECT
        p.product_name,
        p.category_name,
        SUM(f.is_delayed)                                                       AS delay_count,
        COUNT(*)                                                                AS total_orders,
        SUM(f.order_total_value)                                                AS total_value,
        ROUND(SUM(f.is_delayed) * 100.0 / COUNT(*), 2)                         AS delay_pct
    FROM fact_orders f
    JOIN dim_product p ON f.product_card_id = p.product_card_id
    WHERE f.order_status = 'COMPLETE'
    GROUP BY p.product_name, p.category_name
    HAVING COUNT(*) > 50
)
SELECT
    product_name,
    category_name,
    total_orders,
    delay_count,
    delay_pct,
    total_value,
    ROUND(
        SUM(delay_count) OVER (ORDER BY delay_count DESC) * 100.0
        / SUM(delay_count) OVER (), 2
    ) AS cumulative_delay_pct
FROM product_delay
ORDER BY delay_count DESC
LIMIT 20;


-- ──────────────────────────────────────────────────────────
-- Q10: 全局KPI汇总 → Power BI KPI Cards
-- ──────────────────────────────────────────────────────────
SELECT
    'KPI Dashboard'                                                           AS dashboard,
    COUNT(DISTINCT order_id)                                                  AS total_orders,
    ROUND(SUM(on_time) * 100.0 / COUNT(*), 2)                                AS overall_ot_pct,
    ROUND(AVG(days_shipping_real), 1)                                         AS avg_delivery_days,
    ROUND(SUM(order_total_value), 0)                                          AS total_revenue,
    ROUND(SUM(order_item_profit), 0)                                          AS total_profit,
    ROUND(AVG(order_total_value), 2)                                          AS avg_order_value,
    COUNT(DISTINCT customer_id)                                               AS unique_customers,
    ROUND(AVG(CASE WHEN is_delayed = 1 THEN delay_days END), 1)              AS avg_delay_when_late,
    ROUND(SUM(CASE WHEN is_delayed = 1 THEN order_total_value END), 0)       AS revenue_at_risk,
    ROUND(AVG(benefit_per_order), 2)                                          AS avg_benefit_per_order,
    COUNT(DISTINCT s.shipping_mode)                                           AS shipping_modes,
    COUNT(DISTINCT p.category_name)                                           AS product_categories,
    COUNT(DISTINCT c.market)                                                  AS markets_served
FROM fact_orders f
JOIN dim_shipping s ON f.shipping_mode_id = s.shipping_mode_id
JOIN dim_product p  ON f.product_card_id = p.product_card_id
JOIN dim_customer c ON f.customer_id = c.customer_id
WHERE f.order_status = 'COMPLETE';
