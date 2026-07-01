"""Tests for upstream divergence detection."""

import json
import os
import tempfile

import pytest

from agenttrace.annotations.models import Annotation, Judgment
from agenttrace.annotations.store import AnnotationStore
from agenttrace.loopdetect.embeddings import clear_cache
from agenttrace.watchdog.compare import ComparisonResult, FindingType, compare_runs
from agenttrace.watchdog.upstream import (
    UpstreamDivergence,
    diff_field,
    extract_retrieved_context,
    extract_tool_calls,
    find_upstream_divergence,
    get_upstream_steps,
)
from langgraph_replay.storage import NodeExecution, ReplayStorage, Session

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def _load_fixture(name: str) -> dict:
    with open(os.path.join(FIXTURES_DIR, name)) as f:
        return json.load(f)


def _make_executions(fixture: dict, key: str = "executions"):
    """Convert fixture executions into NodeExecution objects."""
    return [
        NodeExecution(
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
        for e in fixture[key]
    ]


class TestGetUpstreamSteps:
    def test_linear_chain(self):
        """1. Returns all steps before the target in a linear chain."""
        from dataclasses import dataclass

        @dataclass
        class FakeExec:
            node_name: str
            execution_order: int

        steps = [
            FakeExec("step_a", 1),
            FakeExec("step_b", 2),
            FakeExec("step_c", 3),
            FakeExec("step_d", 4),
        ]

        upstream = get_upstream_steps(steps, target_execution_order=3)
        assert len(upstream) == 2
        assert [s.execution_order for s in upstream] == [1, 2]

    def test_no_upstream(self):
        """First step has no upstream."""
        from dataclasses import dataclass

        @dataclass
        class FakeExec:
            node_name: str
            execution_order: int

        steps = [FakeExec("step_a", 1)]
        upstream = get_upstream_steps(steps, target_execution_order=1)
        assert len(upstream) == 0

    def test_branch_and_merge(self):
        """Transitive ancestry works with branching graph."""
        from dataclasses import dataclass

        @dataclass
        class FakeExec:
            node_name: str
            execution_order: int

        # Graph: A -> B -> D, A -> C -> D (branch then merge)
        steps = [
            FakeExec("A", 1),
            FakeExec("B", 2),
            FakeExec("C", 3),
            FakeExec("D", 4),
        ]

        upstream = get_upstream_steps(steps, target_execution_order=4)
        assert len(upstream) == 3
        names = [s.node_name for s in upstream]
        assert "A" in names
        assert "B" in names
        assert "C" in names


class TestExtractToolCalls:
    def test_extracts_from_output(self):
        """Extracts tool calls from output_state."""
        from dataclasses import dataclass

        @dataclass
        class FakeExec:
            input_state: str
            output_state: str

        step = FakeExec(
            input_state="{}",
            output_state=json.dumps({"tool_calls": [{"tool_name": "search", "output": "results here"}]}),
        )
        calls = extract_tool_calls(step)
        assert len(calls) == 1
        assert calls[0]["tool_name"] == "search"
        assert calls[0]["output"] == "results here"

    def test_empty_when_no_tools(self):
        """Returns empty list when no tool calls present."""
        from dataclasses import dataclass

        @dataclass
        class FakeExec:
            input_state: str
            output_state: str

        step = FakeExec(input_state="{}", output_state='{"result": "plain"}')
        calls = extract_tool_calls(step)
        assert len(calls) == 0


class TestExtractRetrievedContext:
    def test_extracts_from_output(self):
        """Extracts context entries from output_state."""
        from dataclasses import dataclass

        @dataclass
        class FakeExec:
            output_state: str

        step = FakeExec(
            output_state=json.dumps({
                "retrieved_context": [
                    {"source": "doc1", "content": "Policy text here"},
                    {"source": "doc2", "content": "More policy text"},
                ]
            })
        )
        ctx = extract_retrieved_context(step)
        assert len(ctx) == 2
        assert ctx[0]["source"] == "doc1"
        assert ctx[1]["content"] == "More policy text"

    def test_empty_when_no_context(self):
        """Returns empty list when no context present."""
        from dataclasses import dataclass

        @dataclass
        class FakeExec:
            output_state: str

        step = FakeExec(output_state='{"result": "plain"}')
        ctx = extract_retrieved_context(step)
        assert len(ctx) == 0


class TestDiffField:
    def test_identical_strings(self):
        """Identical strings -> not changed."""
        result = diff_field("hello", "hello", 0.85, "test.field")
        assert result.changed is False
        assert result.similarity_score == 1.0

    def test_similar_strings_above_threshold(self):
        """Similar strings above threshold -> not changed."""
        clear_cache()
        result = diff_field(
            "Order confirmed",
            "Order has been confirmed",
            0.80,  # Low threshold to ensure match
            "test.field",
        )
        assert result.changed is False
        assert result.similarity_score is not None

    def test_different_strings_below_threshold(self):
        """Different strings below threshold -> changed."""
        clear_cache()
        result = diff_field(
            "Refund approved",
            "Refund denied",
            0.85,
            "test.field",
        )
        assert result.changed is True
        assert result.similarity_score is not None
        assert result.similarity_score < 0.85

    def test_non_string_exact_match(self):
        """Non-string values use exact equality."""
        result = diff_field({"key": "val"}, {"key": "val"}, 0.85, "test.field")
        assert result.changed is False

    def test_non_string_mismatch(self):
        """Non-string values that differ -> changed."""
        result = diff_field({"key": "a"}, {"key": "b"}, 0.85, "test.field")
        assert result.changed is True

    def test_missing_in_baseline(self):
        """Value present in new only -> changed."""
        result = diff_field(None, "new value", 0.85, "test.field")
        assert result.changed is True
        assert "new run only" in result.note

    def test_missing_in_new(self):
        """Value present in baseline only -> changed."""
        result = diff_field("old value", None, 0.85, "test.field")
        assert result.changed is True
        assert "baseline run only" in result.note


class TestFindUpstreamDivergence:
    def test_identifies_changed_tool_output(self):
        """6. Correctly identifies changed upstream tool output."""
        clear_cache()
        fixture = _load_fixture("upstream_divergence_fixture.json")
        baseline_execs = _make_executions(fixture, "baseline_executions")
        new_execs = _make_executions(fixture, "new_executions")

        divergences = find_upstream_divergence(
            baseline_executions=baseline_execs,
            new_executions=new_execs,
            regression_step_node_name="summarize_answer",
            regression_step_execution_order=5,
            semantic_threshold=0.80,
        )

        # Should find divergence at lookup_refund_policy
        changed = [d for d in divergences if d.changed]
        assert len(changed) >= 1
        policy_div = next((d for d in changed if d.node_name == "lookup_refund_policy"), None)
        assert policy_div is not None
        assert "policy" in policy_div.field_path.lower() or "tool" in policy_div.category

    def test_excludes_unchanged_steps(self):
        """Unrelated unchanged upstream steps are correctly excluded."""
        clear_cache()
        fixture = _load_fixture("upstream_divergence_fixture.json")
        baseline_execs = _make_executions(fixture, "baseline_executions")
        new_execs = _make_executions(fixture, "new_executions")

        divergences = find_upstream_divergence(
            baseline_executions=baseline_execs,
            new_executions=new_execs,
            regression_step_node_name="summarize_answer",
            regression_step_execution_order=5,
            semantic_threshold=0.80,
        )

        # fetch_order_history and check_inventory should NOT appear in changed divergences
        changed = [d for d in divergences if d.changed]
        changed_nodes = {d.node_name for d in changed}
        assert "fetch_order_history" not in changed_nodes
        assert "check_inventory" not in changed_nodes


class TestCompareRunsUpstream:
    def test_default_no_upstream(self):
        """7. Default (include_upstream_divergence=False) -> empty upstream_divergences."""
        ann_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
        trace_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name

        try:
            ann_store = AnnotationStore(ann_db)
            trace_store = ReplayStorage(trace_db)

            fixture = _load_fixture("upstream_divergence_fixture.json")

            # Load baseline
            s = fixture["baseline_session"]
            trace_store.save_session(Session(
                id=s["id"], agent_name=s["agent_name"], created_at=s["created_at"],
                total_nodes=s["total_nodes"], status=s["status"],
                final_output=s.get("final_output", ""), metadata=s.get("metadata", {})
            ))
            for e in fixture["baseline_executions"]:
                trace_store.save_node_execution(NodeExecution(
                    id=e["id"], session_id=e["session_id"], node_name=e["node_name"],
                    execution_order=e["execution_order"], input_state=e.get("input_state", ""),
                    output_state=e.get("output_state", ""), started_at=e.get("started_at", ""),
                    duration_ms=e.get("duration_ms", 0.0), status=e.get("status", "success"),
                    error_message=e.get("error_message"), llm_calls=e.get("llm_calls", 0)
                ))
            for a in fixture["baseline_annotations"]:
                ann_store.save(Annotation(
                    run_id=a["run_id"], step_id=a["step_id"],
                    judgment=Judgment(a["judgment"]), note=a.get("note"), annotator=a.get("annotator")
                ))

            # Load new run
            s2 = fixture["new_session"]
            trace_store.save_session(Session(
                id=s2["id"], agent_name=s2["agent_name"], created_at=s2["created_at"],
                total_nodes=s2["total_nodes"], status=s2["status"],
                final_output=s2.get("final_output", ""), metadata=s2.get("metadata", {})
            ))
            for e in fixture["new_executions"]:
                trace_store.save_node_execution(NodeExecution(
                    id=e["id"], session_id=e["session_id"], node_name=e["node_name"],
                    execution_order=e["execution_order"], input_state=e.get("input_state", ""),
                    output_state=e.get("output_state", ""), started_at=e.get("started_at", ""),
                    duration_ms=e.get("duration_ms", 0.0), status=e.get("status", "success"),
                    error_message=e.get("error_message"), llm_calls=e.get("llm_calls", 0)
                ))

            # Default mode - no upstream analysis
            result = compare_runs(
                baseline_run_id="upstream-baseline-001",
                new_run_id="upstream-new-001",
                annotation_store=ann_store,
                trace_store=trace_store,
            )

            # Regression should be found
            assert result.has_regression
            reg_finding = next(f for f in result.findings if f.finding_type == FindingType.REGRESSION)
            # But upstream_divergences should be empty
            assert reg_finding.upstream_divergences == []

        finally:
            ann_store.close()
            trace_store.close()
            os.unlink(ann_db)
            os.unlink(trace_db)

    def test_upstream_enabled_populates_divergences(self):
        """8. include_upstream_divergence=True -> regression findings get populated divergences."""
        clear_cache()
        ann_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
        trace_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name

        try:
            ann_store = AnnotationStore(ann_db)
            trace_store = ReplayStorage(trace_db)

            fixture = _load_fixture("upstream_divergence_fixture.json")

            # Load baseline
            s = fixture["baseline_session"]
            trace_store.save_session(Session(
                id=s["id"], agent_name=s["agent_name"], created_at=s["created_at"],
                total_nodes=s["total_nodes"], status=s["status"],
                final_output=s.get("final_output", ""), metadata=s.get("metadata", {})
            ))
            for e in fixture["baseline_executions"]:
                trace_store.save_node_execution(NodeExecution(
                    id=e["id"], session_id=e["session_id"], node_name=e["node_name"],
                    execution_order=e["execution_order"], input_state=e.get("input_state", ""),
                    output_state=e.get("output_state", ""), started_at=e.get("started_at", ""),
                    duration_ms=e.get("duration_ms", 0.0), status=e.get("status", "success"),
                    error_message=e.get("error_message"), llm_calls=e.get("llm_calls", 0)
                ))
            for a in fixture["baseline_annotations"]:
                ann_store.save(Annotation(
                    run_id=a["run_id"], step_id=a["step_id"],
                    judgment=Judgment(a["judgment"]), note=a.get("note"), annotator=a.get("annotator")
                ))

            # Load new run
            s2 = fixture["new_session"]
            trace_store.save_session(Session(
                id=s2["id"], agent_name=s2["agent_name"], created_at=s2["created_at"],
                total_nodes=s2["total_nodes"], status=s2["status"],
                final_output=s2.get("final_output", ""), metadata=s2.get("metadata", {})
            ))
            for e in fixture["new_executions"]:
                trace_store.save_node_execution(NodeExecution(
                    id=e["id"], session_id=e["session_id"], node_name=e["node_name"],
                    execution_order=e["execution_order"], input_state=e.get("input_state", ""),
                    output_state=e.get("output_state", ""), started_at=e.get("started_at", ""),
                    duration_ms=e.get("duration_ms", 0.0), status=e.get("status", "success"),
                    error_message=e.get("error_message"), llm_calls=e.get("llm_calls", 0)
                ))

            # Upstream enabled
            result = compare_runs(
                baseline_run_id="upstream-baseline-001",
                new_run_id="upstream-new-001",
                annotation_store=ann_store,
                trace_store=trace_store,
                include_upstream_divergence=True,
                semantic_threshold=0.80,
            )

            # Regression should be found with upstream divergences
            assert result.has_regression
            reg_finding = next(f for f in result.findings if f.finding_type == FindingType.REGRESSION)
            assert len(reg_finding.upstream_divergences) > 0

            # Should identify the policy change
            changed = [d for d in reg_finding.upstream_divergences if d.changed]
            assert any(d.node_name == "lookup_refund_policy" for d in changed)

        finally:
            ann_store.close()
            trace_store.close()
            os.unlink(ann_db)
            os.unlink(trace_db)
