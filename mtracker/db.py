"""SQLite storage backend for mtracker."""

import sqlite3
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_DB_DIR = Path.home() / ".mtracker"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "mtracker.db"


class Database:
    """SQLite database for experiment tracking."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                project TEXT DEFAULT 'default',
                status TEXT DEFAULT 'running',
                config TEXT DEFAULT '{}',
                tags TEXT DEFAULT '[]',
                started_at REAL NOT NULL,
                ended_at REAL,
                notes TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                step INTEGER,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_metrics_run_name
                ON metrics(run_id, name);
            CREATE INDEX IF NOT EXISTS idx_metrics_step
                ON metrics(run_id, step);
            CREATE INDEX IF NOT EXISTS idx_runs_project
                ON runs(project);
        """)
        self.conn.commit()

    def create_run(self, name: str, project: str = "default",
                   config: Optional[Dict] = None, tags: Optional[List[str]] = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs (name, project, config, tags, started_at) VALUES (?, ?, ?, ?, ?)",
            (name, project, json.dumps(config or {}), json.dumps(tags or []), time.time())
        )
        self.conn.commit()
        return cur.lastrowid

    def end_run(self, run_id: int, status: str = "completed", notes: str = ""):
        self.conn.execute(
            "UPDATE runs SET status=?, ended_at=?, notes=? WHERE id=?",
            (status, time.time(), notes, run_id)
        )
        self.conn.commit()

    def log_metric(self, run_id: int, name: str, value: float,
                   step: Optional[int] = None, timestamp: Optional[float] = None):
        self.conn.execute(
            "INSERT INTO metrics (run_id, step, name, value, timestamp) VALUES (?, ?, ?, ?, ?)",
            (run_id, step, name, value, timestamp or time.time())
        )
        self.conn.commit()

    def log_metrics_batch(self, run_id: int, metrics: List[Tuple[str, float, Optional[int]]]):
        now = time.time()
        self.conn.executemany(
            "INSERT INTO metrics (run_id, name, value, step, timestamp) VALUES (?, ?, ?, ?, ?)",
            [(run_id, name, value, step, now) for name, value, step in metrics]
        )
        self.conn.commit()

    def get_run(self, run_id: int) -> Optional[Dict]:
        row = self.conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row:
            return self._row_to_dict(row)
        return None

    def list_runs(self, project: Optional[str] = None, status: Optional[str] = None,
                  limit: int = 50) -> List[Dict]:
        query = "SELECT * FROM runs WHERE 1=1"
        params = []
        if project:
            query += " AND project=?"
            params.append(project)
        if status:
            query += " AND status=?"
            params.append(status)
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        return [self._row_to_dict(r) for r in self.conn.execute(query, params).fetchall()]

    def get_metrics(self, run_id: int, name: Optional[str] = None) -> List[Dict]:
        if name:
            rows = self.conn.execute(
                "SELECT * FROM metrics WHERE run_id=? AND name=? ORDER BY step",
                (run_id, name)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM metrics WHERE run_id=? ORDER BY step, name",
                (run_id,)
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_latest_metrics(self, run_id: int) -> Dict[str, float]:
        """Get the latest value for each metric name in a run."""
        rows = self.conn.execute("""
            SELECT name, value, step FROM metrics
            WHERE run_id=? AND id IN (
                SELECT MAX(id) FROM metrics WHERE run_id=? GROUP BY name
            )
        """, (run_id, run_id)).fetchall()
        return {r["name"]: {"value": r["value"], "step": r["step"]} for r in rows}

    def get_metric_summary(self, run_id: int, name: str) -> Optional[Dict]:
        row = self.conn.execute("""
            SELECT MIN(value) as min_val, MAX(value) as max_val,
                   AVG(value) as avg_val, COUNT(*) as count,
                   (SELECT value FROM metrics WHERE run_id=? AND name=? ORDER BY step DESC LIMIT 1) as last_val,
                   (SELECT step FROM metrics WHERE run_id=? AND name=? ORDER BY step DESC LIMIT 1) as last_step
            FROM metrics WHERE run_id=? AND name=?
        """, (run_id, name, run_id, name, run_id, name)).fetchone()
        if row and row["count"] > 0:
            return self._row_to_dict(row)
        return None

    def delete_run(self, run_id: int):
        self.conn.execute("DELETE FROM metrics WHERE run_id=?", (run_id,))
        self.conn.execute("DELETE FROM runs WHERE id=?", (run_id,))
        self.conn.commit()

    def _row_to_dict(self, row) -> Dict[str, Any]:
        d = dict(row)
        for key in ("config", "tags"):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
