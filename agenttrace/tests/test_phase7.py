"""Tests for Phase 7 — Wiring, Roadmap, and Demo Artifact."""

import json
import os
import tempfile

import pytest
from click.testing import CliRunner

from agenttrace.annotations.models import Annotation, Judgment
from agenttrace.annotations.store import AnnotationStore
from agenttrace.counterfactual.cli import counterfactual
from agenttrace.watchdog.compare import FindingType, compare_runs
from agenttrace.watchdog.upstream import UpstreamDivergence
from langgraph_replay.storage import NodeExecution, ReplayStorage, Session

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def _load_fixture(name: str) -> dict:
    with open(os.path.join(FIXTURES_DIR, name)) as f:
        return json.load(f)


class TestUpstreamDivergenceSerialization:
    def test_to_counterfactual_input(self):
        """1. to_counterfactual_input produces dict with all required keys."""
        div = UpstreamDivergence(
            step_id="3",
            node_name="lookup_policy",
            category="tool_output",
            field_path="tool_calls[0].output",
            changed=True,
            similarity_score=0.62,
            baseline_value="Full refund within 30 days",
            new_value="Refunds require proof of defect",
            note="Policy changed",
        )
        result = div.to_counterfactual_input()

        assert set(result.keys()) == {
            "step_id", "node_name", "category", "field_path",
            "baseline_value", "new_value",
        }
        assert result["step_id"] == "3"
        assert result["node_name"] == "lookup_policy"
        assert result["baseline_value"] == "Full refund within 30 days"

    def test_round_trip(self):
        """to_counterfactual_input -> from_counterfactual_input preserves data."""
        div = UpstreamDivergence(
            step_id="2",
            node_name="fetch_data",
            category="retrieved_context",
            field_path="retrieved_context[0].content",
            changed=True,
            similarity_score=0.45,
            baseline_value={"source": "db", "content": "original"},
            new_value={"source": "db", "content": "changed"},
            note="Context changed",
        )
        data = div.to_counterfactual_input()
        restored = UpstreamDivergence.from_counterfactual_input(data)

        assert restored.step_id == div.step_id
        assert restored.node_name == div.node_name
        assert restored.category == div.category
        assert restored.field_path == div.field_path
        assert restored.baseline_value == div.baseline_value
        assert restored.new_value == div.new_value
        assert restored.changed is True  # Filled in by from_


class TestFromDivergenceCLI:
    def test_from_divergence_and_manual_flags_errors(self):
        """2. Passing both --from-divergence and manual flags exits 2."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create a minimal divergence file
            div_data = {
                "step_id": "2",
                "node_name": "test_node",
                "category": "tool_output",
                "field_path": "output",
                "baseline_value": "a",
                "new_value": "b",
            }
            with open("div.json", "w") as f:
                json.dump(div_data, f)

            result = runner.invoke(counterfactual, [
                "test", "run-1",
                "--baseline", "base-1",
                "--graph", "test:graph",
                "--thread-id", "thread-1",
                "--from-divergence", "div.json",
                "--step", "2",  # Conflicting manual flag
            ])
            assert result.exit_code == 2
            assert "cannot use" in result.output.lower() or "Error" in result.output

    def test_from_divergence_missing_file_errors(self):
        """--from-divergence with nonexistent file exits with error."""
        runner = CliRunner()
        result = runner.invoke(counterfactual, [
            "test", "run-1",
            "--baseline", "base-1",
            "--graph", "test:graph",
            "--thread-id", "thread-1",
            "--from-divergence", "nonexistent.json",
        ])
        assert result.exit_code != 0

    def test_no_input_flags_errors(self):
        """3. No --from-divergence and no manual flags exits 2."""
        runner = CliRunner()
        result = runner.invoke(counterfactual, [
            "test", "run-1",
            "--baseline", "base-1",
            "--graph", "test:graph",
            "--thread-id", "thread-1",
        ])
        assert result.exit_code == 2


class TestDemoRegressionFixture:
    def test_watchdog_exits_1_exact_mode(self):
        """4. demo_regression fixture: watchdog watch without --semantic exits 1."""
        ann_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
        trace_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name

        try:
            ann_store = AnnotationStore(ann_db)
            trace_store = ReplayStorage(trace_db)

            # Load baseline
            baseline = _load_fixture("demo_regression/baseline_run.json")
            s = baseline["session"]
            trace_store.save_session(Session(
                id=s["id"], agent_name=s["agent_name"], created_at=s["created_at"],
                total_nodes=s["total_nodes"], status=s["status"],
                final_output=s.get("final_output", ""), metadata=s.get("metadata", {})
            ))
            for e in baseline["executions"]:
                trace_store.save_node_execution(NodeExecution(
                    id=e["id"], session_id=e["session_id"], node_name=e["node_name"],
                    execution_order=e["execution_order"], input_state=e.get("input_state", ""),
                    output_state=e.get("output_state", ""), started_at=e.get("started_at", ""),
                    duration_ms=e.get("duration_ms", 0.0), status=e.get("status", "success"),
                    error_message=e.get("error_message"), llm_calls=e.get("llm_calls", 0)
                ))
            for a in baseline["annotations"]:
                ann_store.save(Annotation(
                    run_id=a["run_id"], step_id=a["step_id"],
                    judgment=Judgment(a["judgment"]), note=a.get("note"), annotator=a.get("annotator")
                ))

            # Load regressed run
            regressed = _load_fixture("demo_regression/regressed_run.json")
            s2 = regressed["session"]
            trace_store.save_session(Session(
                id=s2["id"], agent_name=s2["agent_name"], created_at=s2["created_at"],
                total_nodes=s2["total_nodes"], status=s2["status"],
                final_output=s2.get("final_output", ""), metadata=s2.get("metadata", {})
            ))
            for e in regressed["executions"]:
                trace_store.save_node_execution(NodeExecution(
                    id=e["id"], session_id=e["session_id"], node_name=e["node_name"],
                    execution_order=e["execution_order"], input_state=e.get("input_state", ""),
                    output_state=e.get("output_state", ""), started_at=e.get("started_at", ""),
                    duration_ms=e.get("duration_ms", 0.0), status=e.get("status", "success"),
                    error_message=e.get("error_message"), llm_calls=e.get("llm_calls", 0)
                ))

            # Exact mode should flag regression
            result = compare_runs(
                baseline_run_id="demo-baseline-001",
                new_run_id="demo-regressed-001",
                annotation_store=ann_store,
                trace_store=trace_store,
                diff_strategy="exact",
            )
            assert result.has_regression

        finally:
            ann_store.close()
            trace_store.close()
            os.unlink(ann_db)
            os.unlink(trace_db)

    def test_watchdog_still_exits_1_semantic_mode(self):
        """5. demo_regression fixture: watchdog watch with --semantic STILL exits 1 (real regression)."""
        from agenttrace.loopdetect.embeddings import clear_cache
        clear_cache()

        ann_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
        trace_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name

        try:
            ann_store = AnnotationStore(ann_db)
            trace_store = ReplayStorage(trace_db)

            # Load baseline
            baseline = _load_fixture("demo_regression/baseline_run.json")
            s = baseline["session"]
            trace_store.save_session(Session(
                id=s["id"], agent_name=s["agent_name"], created_at=s["created_at"],
                total_nodes=s["total_nodes"], status=s["status"],
                final_output=s.get("final_output", ""), metadata=s.get("metadata", {})
            ))
            for e in baseline["executions"]:
                trace_store.save_node_execution(NodeExecution(
                    id=e["id"], session_id=e["session_id"], node_name=e["node_name"],
                    execution_order=e["execution_order"], input_state=e.get("input_state", ""),
                    output_state=e.get("output_state", ""), started_at=e.get("started_at", ""),
                    duration_ms=e.get("duration_ms", 0.0), status=e.get("status", "success"),
                    error_message=e.get("error_message"), llm_calls=e.get("llm_calls", 0)
                ))
            for a in baseline["annotations"]:
                ann_store.save(Annotation(
                    run_id=a["run_id"], step_id=a["step_id"],
                    judgment=Judgment(a["judgment"]), note=a.get("note"), annotator=a.get("annotator")
                ))

            # Load regressed run
            regressed = _load_fixture("demo_regression/regressed_run.json")
            s2 = regressed["session"]
            trace_store.save_session(Session(
                id=s2["id"], agent_name=s2["agent_name"], created_at=s2["created_at"],
                total_nodes=s2["total_nodes"], status=s2["status"],
                final_output=s2.get("final_output", ""), metadata=s2.get("metadata", {})
            ))
            for e in regressed["executions"]:
                trace_store.save_node_execution(NodeExecution(
                    id=e["id"], session_id=e["session_id"], node_name=e["node_name"],
                    execution_order=e["execution_order"], input_state=e.get("input_state", ""),
                    output_state=e.get("output_state", ""), started_at=e.get("started_at", ""),
                    duration_ms=e.get("duration_ms", 0.0), status=e.get("status", "success"),
                    error_message=e.get("error_message"), llm_calls=e.get("llm_calls", 0)
                ))

            # Semantic mode should ALSO flag regression (this is a real behavior change, not wording)
            result = compare_runs(
                baseline_run_id="demo-baseline-001",
                new_run_id="demo-regressed-001",
                annotation_store=ann_store,
                trace_store=trace_store,
                diff_strategy="semantic",
                semantic_threshold=0.85,
            )
            assert result.has_regression

        finally:
            ann_store.close()
            trace_store.close()
            os.unlink(ann_db)
            os.unlink(trace_db)


class TestRoadmap:
    def test_roadmap_exists(self):
        """ROADMAP.md exists at repo root."""
        # agenttrace/tests/test_phase7.py -> agenttrace/tests -> agenttrace -> repo_root
        repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
        roadmap_path = os.path.join(repo_root, "ROADMAP.md")
        assert os.path.exists(roadmap_path), f"ROADMAP.md not found at {os.path.abspath(roadmap_path)}"

    def test_roadmap_has_required_sections(self):
        """ROADMAP.md contains Shipped, Deliberately Deferred, Explicitly Out of Scope."""
        repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
        roadmap_path = os.path.join(repo_root, "ROADMAP.md")
        with open(roadmap_path) as f:
            content = f.read()

        assert "Shipped" in content
        assert "Deliberately Deferred" in content
        assert "Explicitly Out of Scope" in content
