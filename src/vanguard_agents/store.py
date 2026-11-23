"""Atomic SQLite checkpoints with optimistic concurrency control."""

import json
import sqlite3
from contextlib import contextmanager
from typing import Any

from .types import encode


class ConcurrentUpdate(RuntimeError):
    """A different runner advanced this run."""


class SQLiteStore:
    def __init__(self, path: str) -> None:
        if not path or path == ":memory:":
            raise ValueError("Use a persistent SQLite file")
        self.path = path
        with self._connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, version INTEGER NOT NULL, state TEXT NOT NULL
                )
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _load(self, db, run_id: str) -> tuple[int, dict[str, Any]]:
        row = db.execute(
            "SELECT version, state FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return row[0], json.loads(row[1])

    def start(self, run_id: str, state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        with self._connect() as db:
            db.execute(
                "INSERT INTO runs VALUES (?, 0, ?) ON CONFLICT(id) DO NOTHING",
                (run_id, encode(state)),
            )
            return self._load(db, run_id)

    def load(self, run_id: str) -> tuple[int, dict[str, Any]]:
        with self._connect() as db:
            return self._load(db, run_id)

    def save(self, run_id: str, version: int, state: dict[str, Any]) -> int:
        with self._connect() as db:
            changed = db.execute(
                "UPDATE runs SET version = version + 1, state = ? WHERE id = ? AND version = ?",
                (encode(state), run_id, version),
            ).rowcount
            if not changed:
                raise ConcurrentUpdate(f"Run {run_id!r} advanced concurrently")
        return version + 1
