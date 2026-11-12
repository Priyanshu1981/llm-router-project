"""
database.py
Saves every request to a SQLite database file (logs/requests.db) and
reads it back for the dashboard.

SQLite needs no server: the whole database is one file.
We open a fresh connection for each operation, which is simple and safe
because FastAPI may handle several requests at the same time.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from backend import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at         TEXT    NOT NULL,
    query              TEXT    NOT NULL,
    mode_requested     TEXT    NOT NULL,
    tier               TEXT    NOT NULL,
    model_name         TEXT    NOT NULL,
    route_reason       TEXT    NOT NULL,
    cache_hit          INTEGER NOT NULL DEFAULT 0,
    similarity         REAL,
    answer             TEXT    NOT NULL,
    input_tokens       INTEGER NOT NULL,
    output_tokens      INTEGER NOT NULL,
    latency_ms         REAL    NOT NULL,
    cost_usd           REAL    NOT NULL,
    baseline_cost_usd  REAL    NOT NULL,
    saved_usd          REAL    NOT NULL
)
"""

SUMMARY_SQL = """
SELECT
    COUNT(*)                                                   AS total_requests,
    COALESCE(SUM(cache_hit), 0)                                AS cache_hits,
    COALESCE(SUM(CASE WHEN tier = 'small' THEN 1 ELSE 0 END), 0) AS small_calls,
    COALESCE(SUM(CASE WHEN tier = 'large' THEN 1 ELSE 0 END), 0) AS large_calls,
    COALESCE(SUM(input_tokens), 0)                             AS total_input_tokens,
    COALESCE(SUM(output_tokens), 0)                            AS total_output_tokens,
    COALESCE(SUM(cost_usd), 0)                                 AS total_cost_usd,
    COALESCE(SUM(baseline_cost_usd), 0)                        AS total_baseline_cost_usd,
    COALESCE(SUM(saved_usd), 0)                                AS total_saved_usd,
    COALESCE(AVG(latency_ms), 0)                               AS avg_latency_ms
FROM requests
"""


@contextmanager
def _connect():
    """Open a connection, save changes if everything worked, always close it."""
    connection = sqlite3.connect(config.DB_PATH)
    connection.row_factory = sqlite3.Row  # lets us read columns by name
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db():
    """Create the table if it does not exist yet. Called once when the server starts."""
    with _connect() as connection:
        connection.execute(SCHEMA)


def log_request(record):
    """
    Save one request. record is a dictionary with the keys used below.
    Returns the new row's id.
    """
    row = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query": record["query"],
        "mode_requested": record["mode_requested"],
        "tier": record["tier"],
        "model_name": record["model_name"],
        "route_reason": record["route_reason"],
        "cache_hit": 1 if record.get("cache_hit") else 0,
        "similarity": record.get("similarity"),
        "answer": record["answer"],
        "input_tokens": record["input_tokens"],
        "output_tokens": record["output_tokens"],
        "latency_ms": record["latency_ms"],
        "cost_usd": record["cost_usd"],
        "baseline_cost_usd": record["baseline_cost_usd"],
        "saved_usd": record["saved_usd"],
    }
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO requests (
                created_at, query, mode_requested, tier, model_name, route_reason,
                cache_hit, similarity, answer, input_tokens, output_tokens,
                latency_ms, cost_usd, baseline_cost_usd, saved_usd
            ) VALUES (
                :created_at, :query, :mode_requested, :tier, :model_name, :route_reason,
                :cache_hit, :similarity, :answer, :input_tokens, :output_tokens,
                :latency_ms, :cost_usd, :baseline_cost_usd, :saved_usd
            )
            """,
            row,
        )
        return cursor.lastrowid


def get_recent(limit=20):
    """Return the most recent requests, newest first, as a list of dictionaries."""
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM requests ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    results = []
    for row in rows:
        item = dict(row)
        item["cache_hit"] = bool(item["cache_hit"])
        results.append(item)
    return results


def get_summary():
    """Totals across all logged requests, for the dashboard."""
    with _connect() as connection:
        row = connection.execute(SUMMARY_SQL).fetchone()

    summary = dict(row)
    total = summary["total_requests"]
    baseline = summary["total_baseline_cost_usd"]

    summary["cache_hit_rate"] = round(summary["cache_hits"] / total, 4) if total else 0.0
    if baseline > 0:
        summary["savings_percent"] = round(summary["total_saved_usd"] / baseline * 100.0, 2)
    else:
        summary["savings_percent"] = 0.0

    # Round the money and time values so they look clean in the JSON.
    summary["total_cost_usd"] = round(summary["total_cost_usd"], 10)
    summary["total_baseline_cost_usd"] = round(summary["total_baseline_cost_usd"], 10)
    summary["total_saved_usd"] = round(summary["total_saved_usd"], 10)
    summary["avg_latency_ms"] = round(summary["avg_latency_ms"], 1)
    return summary


def reset_logs():
    """Delete every logged request. Returns how many rows were removed."""
    with _connect() as connection:
        cursor = connection.execute("DELETE FROM requests")
        deleted = cursor.rowcount
        # Restart the id counter at 1 (this table is created with AUTOINCREMENT).
        connection.execute("DELETE FROM sqlite_sequence WHERE name = 'requests'")
    return deleted