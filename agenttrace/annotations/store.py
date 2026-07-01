"""SQLite storage for span annotations."""

import os
import sqlite3
from typing import Optional

from agenttrace.annotations.models import Annotation, Judgment

SCHEMA = """
CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    judgment TEXT NOT NULL CHECK(judgment IN ('correct', 'incorrect', 'expected', 'unexpected')),
    note TEXT,
    annotated_at TEXT NOT NULL,
    annotator TEXT
);

CREATE INDEX IF NOT EXISTS idx_annotations_run_step ON annotations(run_id, step_id);
"""


class AnnotationStore:
    """SQLite-backed annotation storage."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.environ.get(
                "AGENTTRACE_DB",
                os.path.expanduser("~/.langgraph_replay/annotations.db"),
            )
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def save(self, annotation: Annotation, *, allow_overwrite: bool = False) -> Annotation:
        """Persist an annotation.

        Raises ValueError if an annotation already exists for (run_id, step_id)
        and allow_overwrite is False.
        """
        existing = self.get(annotation.run_id, annotation.step_id)
        if existing is not None and not allow_overwrite:
            raise ValueError(
                f"Annotation already exists for run={annotation.run_id!r} step={annotation.step_id!r}. "
                "Use allow_overwrite=True to replace it."
            )

        if existing is not None and allow_overwrite:
            self._conn.execute(
                "UPDATE annotations SET judgment=?, note=?, annotated_at=?, annotator=? WHERE id=?",
                (annotation.judgment.value, annotation.note, annotation.annotated_at, annotation.annotator, existing.id),
            )
            self._conn.commit()
            annotation.id = existing.id
            return annotation

        cur = self._conn.execute(
            "INSERT INTO annotations (run_id, step_id, judgment, note, annotated_at, annotator) VALUES (?,?,?,?,?,?)",
            (annotation.run_id, annotation.step_id, annotation.judgment.value, annotation.note, annotation.annotated_at, annotation.annotator),
        )
        self._conn.commit()
        annotation.id = cur.lastrowid
        return annotation

    def get(self, run_id: str, step_id: str) -> Optional[Annotation]:
        """Retrieve a single annotation by (run_id, step_id)."""
        row = self._conn.execute(
            "SELECT * FROM annotations WHERE run_id=? AND step_id=? ORDER BY annotated_at DESC LIMIT 1",
            (run_id, step_id),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_annotation(row)

    def list_by_run(self, run_id: str, step_id: Optional[str] = None) -> list[Annotation]:
        """List annotations for a run, optionally filtered to one step."""
        if step_id is not None:
            rows = self._conn.execute(
                "SELECT * FROM annotations WHERE run_id=? AND step_id=? ORDER BY step_id, annotated_at",
                (run_id, step_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM annotations WHERE run_id=? ORDER BY step_id, annotated_at",
                (run_id,),
            ).fetchall()
        return [self._row_to_annotation(r) for r in rows]

    def export(self, run_id: str) -> list[dict]:
        """Export all annotations for a run as a list of dicts."""
        return [a.to_dict() for a in self.list_by_run(run_id)]

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_annotation(row: sqlite3.Row) -> Annotation:
        return Annotation(
            id=row["id"],
            run_id=row["run_id"],
            step_id=row["step_id"],
            judgment=Judgment(row["judgment"]),
            note=row["note"],
            annotated_at=row["annotated_at"],
            annotator=row["annotator"],
        )
