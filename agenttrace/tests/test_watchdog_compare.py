"""Tests for the watchdog comparison logic."""

import json
import os
import tempfile

import pytest

from agenttrace.annotations.models import Annotation, Judgment
from agenttrace.annotations.store import AnnotationStore
from agenttrace.watchdog.compare import (
    ComparisonResult,
    FindingType,
    compare_runs,
)
from langgraph_replay.storage import NodeExecution, ReplayStorage, Session

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def _load_fixture(name: str) -> dict:
    with open(os.path.join(FIXTURES_DIR, name)) as f:
        return json.load(f)


def _populate_trace(storage: ReplayStorage, fixture: dict) -> None:
    """Load a fixture's session and executions into a ReplayStorage."""
    s = fixture["session"]
    session = Session(
        id=s["id"],
        agent_name=s["agent_name"],
        created_at=s["created_at"],
        total_nodes=s["total_nodes"],
        status=s["status"],
        final_output=s.get("final_output", ""),
        metadata=s.get("metadata", {}),
    )
    storage.save_session(session)

    for e in fixture["executions"]:
        exec_obj = NodeExecution(
            id=e["id"],
            session_id=e["session_id"],
            node_name=e["node_name"],
            execution_order=e["execution_order"],
            input_state=e.get("input_state", ""),
            output_state=e.get("output_state", ""),
            started_at=e.get("started_at", ""),
            duration_ms=e.get("duration_ms", 0.0),
            status=e.get("status", "success"),
            error_message=e.get("error_message"),
            llm_calls=e.get("llm_calls", 0),
        )
        storage.save_node_execution(exec_obj)


def _populate_annotations(store: AnnotationStore, fixture: dict) -> None:
    """Load a fixture's annotations into an AnnotationStore."""
    for a in fixture.get("annotations", []):
        ann = Annotation(
            run_id=a["run_id"],
            step_id=a["step_id"],
            judgment=Judgment(a["judgment"]),
            note=a.get("note"),
            annotator=a.get("annotator"),
        )
        store.save(ann)


@pytest.fixture
def stores():
    """Create temporary annotation and trace stores, pre-populated with baseline fixture."""
    ann_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    trace_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name

    ann_store = AnnotationStore(ann_db)
    trace_store = ReplayStorage(trace_db)

    # Load baseline
    baseline = _load_fixture("baseline_run_fixture.json")
    _populate_trace(trace_store, baseline)
    _populate_annotations(ann_store, baseline)

    yield ann_store, trace_store

    ann_store.close()
    trace_store.close()
    os.unlink(ann_db)
    os.unlink(trace_db)


class TestCompareRuns:
    def test_clean_run_no_regression(self, stores):
        """1. Clean run (no changes to annotated steps) -> zero regressions, exit 0."""
        ann_store, trace_store = stores

        # Load clean run into trace store
        clean = _load_fixture("clean_run_fixture.json")
        _populate_trace(trace_store, clean)

        result = compare_runs(
            baseline_run_id="baseline-run-001",
            new_run_id="clean-run-001",
            annotation_store=ann_store,
            trace_store=trace_store,
        )

        assert not result.has_regression
        assert result.regression_count == 0
        assert all(f.finding_type == FindingType.UNCHANGED for f in result.findings)

    def test_regressed_run_detected(self, stores):
        """2. Regressed run (one annotated-correct step changed) -> exactly one regression, exit 1."""
        ann_store, trace_store = stores

        regressed = _load_fixture("regressed_run_fixture.json")
        _populate_trace(trace_store, regressed)

        result = compare_runs(
            baseline_run_id="baseline-run-001",
            new_run_id="regressed-run-001",
            annotation_store=ann_store,
            trace_store=trace_store,
        )

        assert result.has_regression
        assert result.regression_count == 1

        regressed_finding = [f for f in result.findings if f.finding_type == FindingType.REGRESSION][0]
        assert regressed_finding.node_name == "node_calculate_total"
        assert "$42.00" in regressed_finding.baseline_output
        assert "$NaN" in regressed_finding.new_output

    def test_no_annotations_exits_with_error(self):
        """3. Baseline with zero correct/expected annotations -> exit code 2 (ValueError)."""
        ann_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
        trace_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name

        ann_store = AnnotationStore(ann_db)
        trace_store = ReplayStorage(trace_db)

        try:
            # No annotations saved
            with pytest.raises(ValueError, match="No ground-truth annotations"):
                compare_runs(
                    baseline_run_id="nonexistent-run",
                    new_run_id="also-nonexistent",
                    annotation_store=ann_store,
                    trace_store=trace_store,
                )
        finally:
            ann_store.close()
            trace_store.close()
            os.unlink(ann_db)
            os.unlink(trace_db)

    def test_structural_change_not_counted_as_regression(self, stores):
        """4. Missing node -> structural_change finding, NOT regression (exit stays 0)."""
        ann_store, trace_store = stores

        missing = _load_fixture("missing_node_run_fixture.json")
        _populate_trace(trace_store, missing)

        result = compare_runs(
            baseline_run_id="baseline-run-001",
            new_run_id="missing-node-run-001",
            annotation_store=ann_store,
            trace_store=trace_store,
        )

        # node_send_confirmation is missing -> structural_change
        struct_findings = [f for f in result.findings if f.finding_type == FindingType.STRUCTURAL_CHANGE]
        assert len(struct_findings) >= 1
        assert any(f.node_name == "node_send_confirmation" for f in struct_findings)

        # But no regression -> exit code 0
        assert not result.has_regression

    def test_incorrect_annotations_ignored(self, stores):
        """5. incorrect/unexpected annotated steps are never evaluated for regression."""
        ann_store, trace_store = stores

        # Save an incorrect annotation for a step that DOES change
        # Use allow_overwrite since baseline already has a correct annotation for this step
        ann_store.save(Annotation(
            run_id="baseline-run-001",
            step_id="node_fetch_price",
            judgment=Judgment.INCORRECT,
            note="This was wrong",
        ), allow_overwrite=True)

        # Load a run where node_fetch_price output differs
        regressed = _load_fixture("regressed_run_fixture.json")
        # Modify fetch_price output to differ from baseline
        regressed["executions"][0]["output_state"] = '{"price": 99.99, "currency": "USD"}'
        _populate_trace(trace_store, regressed)

        result = compare_runs(
            baseline_run_id="baseline-run-001",
            new_run_id="regressed-run-001",
            annotation_store=ann_store,
            trace_store=trace_store,
        )

        # node_fetch_price change should NOT be flagged (annotated incorrect)
        fetch_findings = [f for f in result.findings if f.node_name == "node_fetch_price"]
        assert len(fetch_findings) == 0

    def test_json_report_schema_is_stable(self, stores):
        """5b. JSON report schema has exactly the expected keys (Phase 3/4 lock)."""
        ann_store, trace_store = stores

        clean = _load_fixture("clean_run_fixture.json")
        _populate_trace(trace_store, clean)

        result = compare_runs(
            baseline_run_id="baseline-run-001",
            new_run_id="clean-run-001",
            annotation_store=ann_store,
            trace_store=trace_store,
        )

        from agenttrace.watchdog.report import format_json_report

        report = format_json_report(result)

        # Top-level keys
        expected_top = {"baseline_run_id", "new_run_id", "regression_count", "structural_change_count", "has_regression", "findings"}
        assert set(report.keys()) == expected_top

        # Finding keys
        expected_finding = {"step_id", "node_name", "judgment", "finding_type", "baseline_output", "new_output", "annotation_note"}
        for f in report["findings"]:
            assert set(f.keys()) == expected_finding
