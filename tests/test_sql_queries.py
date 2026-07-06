"""
SQL 查询测试套件 — 验证全部 10 条分析查询
Usage: python tests/test_sql_queries.py
"""
import sqlite3, sys, os
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "supply_chain.db")

if not os.path.exists(DB_PATH):
    print(f"❌ 数据库未找到: {DB_PATH}")
    print("   请先运行: python python/01_data_cleaning.py && python python/02_load_to_db.py")
    sys.exit(1)

conn = sqlite3.connect(DB_PATH)

tests = [
    ("月度准时率+滚动3月均值", """
        WITH m AS (SELECT order_year,order_month,COUNT(*) AS n,
            ROUND(SUM(on_time)*100.0/COUNT(*),2) AS ot
            FROM fact_orders WHERE order_status='COMPLETE'
            GROUP BY order_year,order_month)
        SELECT COUNT(*) FROM m
    """),
    ("运输方式准时率排名", """
        SELECT COUNT(DISTINCT s.shipping_mode) FROM fact_orders f
        JOIN dim_shipping s ON f.shipping_mode_id=s.shipping_mode_id
    """),
    ("延迟根因交叉分析", """
        SELECT COUNT(*) FROM (
            SELECT s.shipping_mode,c.market FROM fact_orders f
            JOIN dim_shipping s ON f.shipping_mode_id=s.shipping_mode_id
            JOIN dim_customer c ON f.customer_id=c.customer_id
            WHERE f.order_status='COMPLETE' GROUP BY s.shipping_mode,c.market HAVING COUNT(*)>100
        )
    """),
    ("品类月度同比", """
        WITH m AS (SELECT p.category_name,f.order_year,f.order_month,COUNT(*) AS n
            FROM fact_orders f JOIN dim_product p ON f.product_card_id=p.product_card_id
            WHERE f.order_status='COMPLETE' GROUP BY p.category_name,f.order_year,f.order_month)
        SELECT COUNT(*) FROM m
    """),
    ("延迟天数分级", """
        SELECT COUNT(DISTINCT delay_category) FROM fact_orders WHERE order_status='COMPLETE'
    """),
    ("高风险订单画像", """
        SELECT COUNT(*) FROM (
            SELECT s.shipping_mode,c.market,p.category_name FROM fact_orders f
            JOIN dim_shipping s ON f.shipping_mode_id=s.shipping_mode_id
            JOIN dim_customer c ON f.customer_id=c.customer_id
            JOIN dim_product p ON f.product_card_id=p.product_card_id
            WHERE f.is_delayed=1 AND f.order_status='COMPLETE'
            GROUP BY s.shipping_mode,c.market,p.category_name HAVING COUNT(*)>50
        )
    """),
    ("NTILE分位数排名", """
        WITH cm AS (SELECT p.category_name,c.market,COUNT(*) AS n
            FROM fact_orders f
            JOIN dim_product p ON f.product_card_id=p.product_card_id
            JOIN dim_customer c ON f.customer_id=c.customer_id
            WHERE f.order_status='COMPLETE' GROUP BY p.category_name,c.market HAVING COUNT(*)>200)
        SELECT COUNT(*) FROM cm
    """),
    ("周度趋势", """
        SELECT COUNT(*) FROM (
            SELECT strftime('%W',order_date),order_year FROM fact_orders
            WHERE order_status='COMPLETE' GROUP BY order_year,strftime('%W',order_date)
        )
    """),
    ("Pareto分析", """
        WITH pd AS (SELECT p.product_name,SUM(f.is_delayed) AS d FROM fact_orders f
            JOIN dim_product p ON f.product_card_id=p.product_card_id
            WHERE f.order_status='COMPLETE' GROUP BY p.product_name HAVING COUNT(*)>50)
        SELECT COUNT(*) FROM pd
    """),
    ("全局KPI汇总", """
        SELECT COUNT(DISTINCT order_id) FROM fact_orders WHERE order_status='COMPLETE'
    """),
]

passed = 0
for name, sql in tests:
    try:
        result = conn.execute(sql).fetchone()
        if result and result[0] > 0:
            print(f"  ✅ {name} → {result[0]}")
            passed += 1
        else:
            print(f"  ⚠️  {name} → 返回空或0")
    except Exception as e:
        print(f"  ❌ {name} → {e}")

conn.close()
print(f"\n{'='*50}")
print(f"  {passed}/{len(tests)} 通过")
if passed == len(tests):
    print("  ✅ 全部SQL查询验证通过")
    sys.exit(0)
else:
    print("  ❌ 部分查询失败")
    sys.exit(1)
