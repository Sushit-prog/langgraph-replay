"""Tests for the DiagnosisEngine."""

import json
from unittest.mock import MagicMock, patch

import pytest

from langgraph_replay.blame import BlameResult
from langgraph_replay.diagnosis import DiagnosisEngine, DiagnosisResult
from langgraph_replay.storage import NodeExecution, Session


class TestDiagnosisEngine:
    """Tests for LLM-based diagnosis."""

    def test_diagnosis_returns_result(self, storage):
        """Mock LLM call to return valid JSON, assert DiagnosisResult fields."""
        session = Session(
            id="session_diag_test",
            agent_name="test",
            created_at="2024-01-15T10:00:00Z",
            total_nodes=2,
            status="failed",
        )
        storage.save_session(session)
        storage.save_node_execution(NodeExecution(
            session_id="session_diag_test",
            node_name="node_1",
            execution_order=0,
            input_state='{"a": 1}',
            output_state='{"a": 1}',
            started_at="2024-01-15T10:00:00Z",
            duration_ms=10.0,
            status="success",
        ))
        storage.save_node_execution(NodeExecution(
            session_id="session_diag_test",
            node_name="node_2",
            execution_order=1,
            input_state='{"a": 1}',
            output_state="{}",
            started_at="2024-01-15T10:00:01Z",
            duration_ms=5.0,
            status="error",
            error_message="KeyError: missing",
        ))

        blamed = NodeExecution(
            session_id="session_diag_test",
            node_name="node_2",
            execution_order=1,
            input_state='{"a": 1}',
            output_state="{}",
            started_at="2024-01-15T10:00:01Z",
            duration_ms=5.0,
            status="error",
            error_message="KeyError: missing",
        )
        blame_result = BlameResult(
            blamed_node=blamed,
            reason="Node raised error",
            confidence="high",
        )

        mock_response = {
            "root_cause": "The node tries to access a key that was never added to state.",
            "fix_suggestions": [
                "Check the upstream node that should populate the key",
                "Add a default value for the missing key",
            ],
            "confidence": "high",
        }

        engine = DiagnosisEngine("session_diag_test", storage)
        with patch.object(engine, "_call_llm", return_value=mock_response):
            result = engine.diagnose(blame_result)

        assert isinstance(result, DiagnosisResult)
        assert result.root_cause == mock_response["root_cause"]
        assert len(result.fix_suggestions) == 2
        assert result.confidence == "high"
        assert result.blamed_node_name == "node_2"

    def test_diagnosis_handles_no_blamed_node(self, storage):
        """Pass BlameResult with blamed_node=None, assert defaults."""
        engine = DiagnosisEngine("session_none", storage)
        blame_result = BlameResult(blamed_node=None, reason="No issues found")

        result = engine.diagnose(blame_result)

        assert result.root_cause == "No issues detected"
        assert result.fix_suggestions == []
        assert result.blamed_node_name is None

    def test_diagnosis_handles_llm_failure(self, storage):
        """Mock LLM to raise exception, assert graceful fallback."""
        session = Session(
            id="session_fail_diag",
            agent_name="test",
            created_at="2024-01-15T10:00:00Z",
            total_nodes=1,
            status="failed",
        )
        storage.save_session(session)
        storage.save_node_execution(NodeExecution(
            session_id="session_fail_diag",
            node_name="node_1",
            execution_order=0,
            input_state='{"a": 1}',
            output_state="{}",
            started_at="2024-01-15T10:00:00Z",
            duration_ms=5.0,
            status="error",
            error_message="RuntimeError",
        ))

        blamed = NodeExecution(
            session_id="session_fail_diag",
            node_name="node_1",
            execution_order=0,
            input_state='{"a": 1}',
            output_state="{}",
            started_at="2024-01-15T10:00:00Z",
            duration_ms=5.0,
            status="error",
            error_message="RuntimeError",
        )
        blame_result = BlameResult(
            blamed_node=blamed,
            reason="error",
            confidence="high",
        )

        engine = DiagnosisEngine("session_fail_diag", storage)
        with patch.object(engine, "_call_llm", side_effect=RuntimeError("API down")):
            result = engine.diagnose(blame_result)

        assert result.root_cause == "Diagnosis unavailable"
        assert result.fix_suggestions == []
        assert result.confidence == "low"

    def test_diagnosis_builds_prompt_with_state(self, storage):
        """Assert prompt contains node name, input state, output state."""
        engine = DiagnosisEngine("session_prompt_test", storage)

        blamed = NodeExecution(
            session_id="session_prompt_test",
            node_name="summarize",
            execution_order=1,
            input_state='{"context": "some text", "topic": "AI"}',
            output_state='{"summary": "AI is great"}',
            started_at="2024-01-15T10:00:00Z",
            duration_ms=42.5,
            status="success",
        )
        from langgraph_replay.diff import compute_state_diff
        input_s = {"context": "some text", "topic": "AI"}
        output_s = {"summary": "AI is great"}
        diff = compute_state_diff(input_s, output_s)

        prompt = engine._build_prompt(blamed, diff, "Key 'context' dropped")

        assert "summarize" in prompt
        assert "context" in prompt
        assert "some text" in prompt
        assert "summary" in prompt
        assert "AI is great" in prompt

    def test_diagnosis_includes_source_code(self, storage):
        """Prompt includes source code when graph_nodes provided."""
        def my_node(state):
            return {"result": state.get("x", 0) + 1}

        session = Session(
            id="session_src",
            agent_name="test",
            created_at="2024-01-15T10:00:00Z",
            total_nodes=1,
            status="failed",
        )
        storage.save_session(session)

        engine = DiagnosisEngine(
            "session_src",
            storage,
            graph_nodes={"my_node": my_node},
        )

        blamed = NodeExecution(
            session_id="session_src",
            node_name="my_node",
            execution_order=0,
            input_state='{"x": 1}',
            output_state='{}',
            started_at="2024-01-15T10:00:00Z",
            duration_ms=5.0,
            status="error",
            error_message=" KeyError",
        )
        from langgraph_replay.diff import compute_state_diff
        diff = compute_state_diff({"x": 1}, {})
        prompt = engine._build_prompt(blamed, diff, "error")
        assert "source code" in prompt.lower() or "def " in prompt

    def test_diagnosis_handles_missing_source(self, storage):
        """Built-in function source cannot be read, returns None gracefully."""
        engine = DiagnosisEngine(
            "session_miss",
            storage,
            graph_nodes={"builtin": len},
        )
        result = engine._get_node_source("builtin")
        assert result is None

    def test_diagnosis_without_graph_nodes(self, storage):
        """Prompt says source not available when graph_nodes is None."""
        engine = DiagnosisEngine("session_nog", storage)
        blamed = NodeExecution(
            session_id="session_nog",
            node_name="node_1",
            execution_order=0,
            input_state='{"a": 1}',
            output_state='{}',
            started_at="2024-01-15T10:00:00Z",
            duration_ms=5.0,
            status="error",
        )
        from langgraph_replay.diff import compute_state_diff
        diff = compute_state_diff({"a": 1}, {})
        prompt = engine._build_prompt(blamed, diff, "error")
        assert "Source code not available" in prompt
