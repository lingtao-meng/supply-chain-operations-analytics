"""KPI assertion tests — do the dashboard numbers actually mean what we say?

`test_sql_queries.py` only checks that the ten queries *run*.  This file checks
that the numbers they return are internally consistent and match the claims in
the README:

    python tests/test_kpis.py        # no pytest required
    pytest tests/test_kpis.py -q     # or as a normal test module
"""

from __future__ import annotations

import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "supply_chain.db")

try:  # optional: the file is runnable without pytest
    import pytest

    @pytest.fixture(scope="module")
    def db():
        if not os.path.exists(DB_PATH):
            pytest.skip(f"database not found: {DB_PATH}")
        conn = sqlite3.connect(DB_PATH)
        yield conn
        conn.close()

except ImportError:  # pragma: no cover
    pytest = None


def scalar(conn, sql, params=()):
    row = conn.execute(sql, params).fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------- integrity


def test_fact_not_empty(db):
    assert scalar(db, "SELECT COUNT(*) FROM fact_orders") > 100_000


def test_fact_grain_is_one_row_per_order_line(db):
    """The fact table is at order-line grain: several rows share an order_id."""
    dupes = scalar(
        db,
        """SELECT COUNT(*) FROM (
               SELECT fact_id FROM fact_orders GROUP BY fact_id HAVING COUNT(*) > 1
           )""",
    )
    assert dupes == 0, "fact_id must be unique"
    lines_per_order = scalar(
        db,
        """SELECT AVG(n) FROM (SELECT COUNT(*) AS n FROM fact_orders GROUP BY order_id)""",
    )
    assert 1.0 <= lines_per_order < 5.0, f"avg lines/order = {lines_per_order:.2f}"


def test_delay_days_matches_shipping_dates(db):
    """delay_days = actual - scheduled (no sign flips in the ETL)."""
    bad = scalar(
        db,
        """SELECT COUNT(*) FROM fact_orders
           WHERE ABS(delay_days - (days_shipping_real - days_shipping_scheduled)) > 1e-6""",
    )
    assert bad == 0


def test_on_time_flag_matches_delay(db):
    """on_time = 1 exactly when delay_days <= 0."""
    mismatched = scalar(
        db,
        """SELECT COUNT(*) FROM fact_orders
           WHERE (on_time = 1) != (delay_days <= 0)""",
    )
    assert mismatched == 0


# ---------------------------------------------------------------- KPI range


def test_on_time_pct_within_bounds(db):
    rows = db.execute(
        """SELECT s.shipping_mode,
                  SUM(f.on_time) * 100.0 / COUNT(*)
           FROM fact_orders f JOIN dim_shipping s ON f.shipping_mode_id = s.shipping_mode_id
           WHERE f.order_status = 'COMPLETE'
           GROUP BY s.shipping_mode"""
    ).fetchall()
    assert rows, "no shipping modes returned"
    for mode, pct in rows:
        assert 0 <= pct <= 100, f"{mode}: {pct}"


def test_market_on_time_spread_is_small(db):
    """Markets barely differ — the README claims this, so assert it.

    If a future data refresh makes LATAM genuinely worse, this test should fail
    and the README should be rewritten instead of silently drifting.
    """
    rows = db.execute(
        """SELECT c.market, AVG(f.on_time) * 100
           FROM fact_orders f JOIN dim_customer c ON f.customer_id = c.customer_id
           WHERE f.order_status = 'COMPLETE'
           GROUP BY c.market"""
    ).fetchall()
    pcts = [p for _, p in rows]
    assert len(pcts) >= 4
    assert max(pcts) - min(pcts) < 5, f"market spread too large: {sorted(rows)}"


# ---------------------------------------------------------------- documented claims


def test_first_class_promise_is_unreachable(db):
    """README: First Class promises 1 day, takes 2, on-time ≈ 0%."""
    row = db.execute(
        """SELECT AVG(f.on_time) * 100, AVG(f.days_shipping_scheduled), AVG(f.days_shipping_real)
           FROM fact_orders f JOIN dim_shipping s ON f.shipping_mode_id = s.shipping_mode_id
           WHERE s.shipping_mode = 'First Class' AND f.order_status = 'COMPLETE'"""
    ).fetchone()
    on_time, scheduled, actual = row
    assert on_time < 1.0, f"First Class on-time {on_time:.1f}%"
    assert abs(scheduled - 1.0) < 0.01
    assert actual > scheduled, "actual should exceed the promise"


def test_standard_class_is_the_most_reliable(db):
    """README: Standard Class is the only mode that meets its promise."""
    rows = dict(
        db.execute(
            """SELECT s.shipping_mode, AVG(f.on_time) * 100
               FROM fact_orders f JOIN dim_shipping s ON f.shipping_mode_id = s.shipping_mode_id
               WHERE f.order_status = 'COMPLETE'
               GROUP BY s.shipping_mode"""
        ).fetchall()
    )
    assert rows["Standard Class"] == max(rows.values())


def test_majority_of_delays_are_short(db):
    """README: more than half of the late orders are only 1-3 days late."""
    total, short = db.execute(
        """SELECT SUM(is_delayed),
                  SUM(CASE WHEN delay_days BETWEEN 1 AND 3 THEN 1 ELSE 0 END)
           FROM fact_orders WHERE order_status = 'COMPLETE'"""
    ).fetchone()
    assert total > 0
    assert short / total > 0.5, f"{short / total:.1%} of delays are 1-3 days"


if __name__ == "__main__":
    import sys

    conn = sqlite3.connect(DB_PATH)
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            if fn.__code__.co_argcount == 1:
                fn(conn)
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    conn.close()
    print(f"\n{len([k for k in globals() if k.startswith('test_')]) - failures} passed, {failures} failed")
    sys.exit(1 if failures else 0)
