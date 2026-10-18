"""SQL syntax test — runs the DDL and every analytics query against an empty
in-memory database.

The committed database is not in the repo (too large), so this test cannot check
KPI values.  What it *can* check in CI is that the schema and the ten queries
stay syntactically valid and reference only existing columns — the failure mode
that actually breaks dashboards after a refactor.
"""

from __future__ import annotations

import os
import sqlite3

SQL_DIR = os.path.join(os.path.dirname(__file__), "..", "sql")


def _read(name: str) -> str:
    with open(os.path.join(SQL_DIR, name), encoding="utf-8") as fh:
        return fh.read()


def test_schema_is_valid_sql():
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(_read("01_create_schema.sql"))
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "fact_orders" in tables
        for dim in ("dim_customer", "dim_product", "dim_date", "dim_shipping"):
            assert dim in tables, f"missing {dim}"
    finally:
        conn.close()


def test_analytics_queries_run_on_empty_tables():
    """Every statement must parse and resolve columns (returns 0 rows)."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(_read("01_create_schema.sql"))
        sql = _read("02_analytics_queries.sql")
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        assert len(statements) >= 10, f"only {len(statements)} queries found"
        for i, stmt in enumerate(statements, 1):
            try:
                conn.execute(stmt).fetchall()
            except sqlite3.Error as exc:  # pragma: no cover - CI signal
                raise AssertionError(f"query #{i} failed: {exc}\n{stmt[:200]}") from exc
    finally:
        conn.close()


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{2 - failures if failures <= 2 else 0} passed, {failures} failed")
    sys.exit(1 if failures else 0)
