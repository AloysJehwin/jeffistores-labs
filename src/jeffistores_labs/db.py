"""DB helpers for querying the local jeffi_replica Postgres.

Designed for use from notebooks on the Razer where Postgres lives on
localhost:5432. Configuration is read from environment first, falling back
to the standard ~/.pgpass entry created by the bootstrap.

Usage from a notebook:

    from jeffistores_labs.db import query, engine, list_tables
    df = query("SELECT id, name, price FROM products LIMIT 10")
    list_tables()
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Engine, create_engine, text


def _read_pgpass(host: str, port: int, db: str, user: str) -> str | None:
    """Look up password from ~/.pgpass — same format psql uses."""
    pgpass = Path(os.environ.get("PGPASSFILE", Path.home() / ".pgpass"))
    if not pgpass.exists():
        return None
    for raw in pgpass.read_text().splitlines():
        if not raw or raw.startswith("#"):
            continue
        try:
            h, p, d, u, pw = raw.split(":", 4)
        except ValueError:
            continue
        if (h in (host, "*")) and (p in (str(port), "*")) and (d in (db, "*")) and (u in (user, "*")):
            return pw
    return None


def _resolve_url() -> str:
    """Build a SQLAlchemy URL for the local jeffi_replica."""
    if explicit := os.environ.get("JEFFI_REPLICA_URL"):
        return explicit

    host = os.environ.get("JEFFI_REPLICA_HOST", "localhost")
    port = int(os.environ.get("JEFFI_REPLICA_PORT", "5432"))
    db = os.environ.get("JEFFI_REPLICA_DB", "jeffi_replica")
    user = os.environ.get("JEFFI_REPLICA_USER", "jeffi_replica")
    pw = os.environ.get("JEFFI_REPLICA_PASSWORD") or _read_pgpass(host, port, db, user)
    if pw is None:
        raise RuntimeError(
            "No password for jeffi_replica found. Set JEFFI_REPLICA_PASSWORD or "
            "ensure ~/.pgpass has an entry for "
            f"{host}:{port}:{db}:{user}:<password>"
        )
    return f"postgresql+psycopg://{user}:{pw}@{host}:{port}/{db}"


@lru_cache(maxsize=1)
def engine() -> Engine:
    """Cached SQLAlchemy engine for the replica."""
    return create_engine(_resolve_url(), pool_pre_ping=True, future=True)


def query(sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame."""
    with engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def list_tables(schema: str = "public") -> pd.DataFrame:
    """All tables in the schema with row estimates."""
    return query(
        """
        SELECT s.relname AS table,
               s.n_live_tup AS rows_est,
               pg_size_pretty(pg_total_relation_size(c.oid)) AS size
        FROM pg_stat_user_tables s
        JOIN pg_class c ON c.oid = s.relid
        WHERE s.schemaname = :schema
        ORDER BY s.n_live_tup DESC
        """,
        {"schema": schema},
    )


def describe(table: str, schema: str = "public") -> pd.DataFrame:
    """Column list with types for a table."""
    return query(
        """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
        ORDER BY ordinal_position
        """,
        {"schema": schema, "table": table},
    )
