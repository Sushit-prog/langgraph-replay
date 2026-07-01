"""Tests for the annotation store."""

import json
import os
import tempfile

import pytest

from agenttrace.annotations.models import Annotation, Judgment
from agenttrace.annotations.store import AnnotationStore


@pytest.fixture
def store():
    """Create a fresh temporary store for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    s = AnnotationStore(db_path)
    yield s
    s.close()
    os.unlink(db_path)


class TestAnnotationStore:
    def test_write_and_read_back(self, store: AnnotationStore):
        """1. Writing an annotation and reading it back matches exactly."""
        ann = Annotation(
            run_id="run-001",
            step_id="step-1",
            judgment=Judgment.CORRECT,
            note="Looks good",
            annotator="tester",
        )
        saved = store.save(ann)

        assert saved.id is not None
        fetched = store.get("run-001", "step-1")
        assert fetched is not None
        assert fetched.run_id == "run-001"
        assert fetched.step_id == "step-1"
        assert fetched.judgment == Judgment.CORRECT
        assert fetched.note == "Looks good"
        assert fetched.annotator == "tester"

    def test_invalid_judgment_raises(self):
        """2. Writing an annotation with an invalid judgment value raises a validation error."""
        with pytest.raises(ValueError, match="is not a valid Judgment"):
            Judgment("bogus")

    def test_overwrite_prevention(self, store: AnnotationStore):
        """3. Writing a second annotation for the same (run_id, step_id) without confirmation raises."""
        ann1 = Annotation(
            run_id="run-002",
            step_id="step-2",
            judgment=Judgment.CORRECT,
            note="First",
        )
        store.save(ann1)

        ann2 = Annotation(
            run_id="run-002",
            step_id="step-2",
            judgment=Judgment.INCORRECT,
            note="Second",
        )
        with pytest.raises(ValueError, match="already exists"):
            store.save(ann2, allow_overwrite=False)

        # Original is untouched
        fetched = store.get("run-002", "step-2")
        assert fetched.judgment == Judgment.CORRECT
        assert fetched.note == "First"

    def test_overwrite_allowed(self, store: AnnotationStore):
        """Overwrite=True replaces the existing annotation."""
        ann1 = Annotation(run_id="r1", step_id="s1", judgment=Judgment.CORRECT, note="old")
        store.save(ann1)

        ann2 = Annotation(run_id="r1", step_id="s1", judgment=Judgment.INCORRECT, note="new")
        store.save(ann2, allow_overwrite=True)

        fetched = store.get("r1", "s1")
        assert fetched.judgment == Judgment.INCORRECT
        assert fetched.note == "new"

    def test_list_filters_by_run(self, store: AnnotationStore):
        """4. list filters correctly by run."""
        store.save(Annotation(run_id="r1", step_id="s1", judgment=Judgment.CORRECT))
        store.save(Annotation(run_id="r1", step_id="s2", judgment=Judgment.INCORRECT))
        store.save(Annotation(run_id="r2", step_id="s1", judgment=Judgment.EXPECTED))

        results = store.list_by_run("r1")
        assert len(results) == 2
        assert all(a.run_id == "r1" for a in results)

    def test_list_filters_by_run_and_step(self, store: AnnotationStore):
        """4b. list filters correctly by run+step."""
        store.save(Annotation(run_id="r1", step_id="s1", judgment=Judgment.CORRECT))
        store.save(Annotation(run_id="r1", step_id="s2", judgment=Judgment.INCORRECT))

        results = store.list_by_run("r1", step_id="s2")
        assert len(results) == 1
        assert results[0].step_id == "s2"

    def test_export_produces_valid_json(self, store: AnnotationStore):
        """5. export produces valid JSON matching the stored records exactly."""
        store.save(Annotation(run_id="r1", step_id="s1", judgment=Judgment.CORRECT, note="ok"))
        store.save(Annotation(run_id="r1", step_id="s2", judgment=Judgment.INCORRECT, note="bad"))

        data = store.export("r1")
        assert len(data) == 2

        # Verify JSON key names match the spec
        expected_keys = {"id", "run_id", "step_id", "judgment", "note", "annotated_at", "annotator"}
        for record in data:
            assert set(record.keys()) == expected_keys

        # Verify round-trip through JSON
        json_str = json.dumps(data, indent=2)
        parsed = json.loads(json_str)
        assert parsed == data

    def test_idempotent_init(self, tmp_path):
        """6. Store initializes cleanly on first run and doesn't error on second run."""
        db_path = str(tmp_path / "test.db")
        s1 = AnnotationStore(db_path)
        s1.save(Annotation(run_id="r1", step_id="s1", judgment=Judgment.CORRECT))
        s1.close()

        # Second open should not raise
        s2 = AnnotationStore(db_path)
        results = s2.list_by_run("r1")
        assert len(results) == 1
        s2.close()
