"""Annotation data models."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class Judgment(str, Enum):
    """Allowed annotation judgments."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    EXPECTED = "expected"
    UNEXPECTED = "unexpected"


@dataclass
class Annotation:
    """A single annotation attached to a step/span."""

    run_id: str
    step_id: str
    judgment: Judgment
    note: Optional[str] = None
    id: Optional[int] = None
    annotated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    annotator: Optional[str] = None

    def to_dict(self) -> dict:
        """Export as dict for JSON serialization."""
        return {
            "id": self.id,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "judgment": self.judgment.value,
            "note": self.note,
            "annotated_at": self.annotated_at,
            "annotator": self.annotator,
        }
