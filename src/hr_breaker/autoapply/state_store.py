"""Local dedup/audit store for the auto-apply pipeline (SQLite, one row per vacancy seen)."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

DEFAULT_DB_PATH = Path(".cache/autoapply.sqlite3")

# "seen" alone means a run recorded the vacancy but was interrupted before
# finishing it (tailoring, applying) - it should be retried, not skipped.
_RESOLVED_STATUSES = {"ready", "applied", "failed", "skipped"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS applications (
    vacancy_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,            -- seen | tailored | ready | applied | skipped | failed
    title TEXT,
    company TEXT,
    url TEXT,
    trigger_keyword TEXT,
    cover_letter TEXT,
    pdf_path TEXT,
    error TEXT,
    seen_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    raw_json TEXT
);
"""


class AutoApplyStore:
    """Tracks which vacancies have already been processed, to avoid duplicate work/applications."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def seen(self, vacancy_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM applications WHERE vacancy_id = ?", (vacancy_id,)
            ).fetchone()
        return row is not None

    def is_resolved(self, vacancy_id: str) -> bool:
        """True if this vacancy reached a terminal state - a bare "seen" row (recorded
        but interrupted before tailoring/applying finished) is not resolved and should
        be retried."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM applications WHERE vacancy_id = ?", (vacancy_id,)
            ).fetchone()
        return row is not None and row[0] in _RESOLVED_STATUSES

    def get(self, vacancy_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM applications WHERE vacancy_id = ?", (vacancy_id,)
            ).fetchone()
        return dict(row) if row else None

    def upsert(
        self,
        vacancy_id: str,
        status: str,
        *,
        title: str | None = None,
        company: str | None = None,
        url: str | None = None,
        trigger_keyword: str | None = None,
        cover_letter: str | None = None,
        pdf_path: str | None = None,
        error: str | None = None,
        raw: dict | None = None,
    ) -> None:
        now = datetime.now().isoformat()
        existing = self.get(vacancy_id)
        with self._connect() as conn:
            if existing is None:
                conn.execute(
                    """INSERT INTO applications
                       (vacancy_id, status, title, company, url, trigger_keyword,
                        cover_letter, pdf_path, error, seen_at, updated_at, raw_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        vacancy_id, status, title, company, url, trigger_keyword,
                        cover_letter, pdf_path, error, now, now,
                        json.dumps(raw) if raw else None,
                    ),
                )
            else:
                conn.execute(
                    """UPDATE applications SET
                        status = ?, title = COALESCE(?, title), company = COALESCE(?, company),
                        url = COALESCE(?, url), trigger_keyword = COALESCE(?, trigger_keyword),
                        cover_letter = COALESCE(?, cover_letter), pdf_path = COALESCE(?, pdf_path),
                        error = ?, updated_at = ?
                       WHERE vacancy_id = ?""",
                    (status, title, company, url, trigger_keyword, cover_letter, pdf_path,
                     error, now, vacancy_id),
                )

    def list_by_status(self, status: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM applications WHERE status = ? ORDER BY updated_at DESC", (status,)
            ).fetchall()
        return [dict(r) for r in rows]

    def count_applied_since(self, since_iso: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM applications WHERE status = 'applied' AND updated_at >= ?",
                (since_iso,),
            ).fetchone()
        return row[0] if row else 0
