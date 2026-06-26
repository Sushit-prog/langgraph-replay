"""Tests for the BlameEngine."""

from unittest.mock import MagicMock, patch

import pytest

from langgraph_replay.blame import BlameEngine, BlameResult
from langgraph_replay.storage import NodeExecution, Session


class TestBlameEngine:
    """Tests for blame analysis."""

    def test_blame_identifies_error_node(self, storage, sample_executions):
        """Error node should be identified as the blamed node."""
        session = Session(
            id="session_test01",
            agent_name="test",
            created_at="2024-01-15T10:30:00Z",
            total_nodes=5,
            status="failed",
        )
        storage.save_session(session)
        for exec in sample_executions:
            storage.save_node_execution(exec)

        engine = BlameEngine("session_test01", storage)
        result = engine.run()

        assert result.blamed_node is not None
        assert result.blamed_node.node_name == "node_3"
        assert result.confidence == "high"

    def test_blame_identifies_missing_key(self, storage):
        """Node that permanently drops a key should be blamed."""
        session = Session(
            id="session_key_test",
            agent_name="test",
            created_at="2024-01-15T10:30:00Z",
            total_nodes=2,
            status="completed",
        )
        storage.save_session(session)

        # Initial state has keys a, b, c. Final output only has a.
        # Node 2 is the last to see b and c in input but not output,
        # and they never reappear.
        execs = [
            NodeExecution(
                session_id="session_key_test",
                node_name="node_1",
                execution_order=0,
                input_state='{"a": 1, "b": 2, "c": 3}',
                output_state='{"a": 1, "b": 2, "c": 3, "d": 4}',
                started_at="2024-01-15T10:30:00Z",
                duration_ms=10.0,
                status="success",
            ),
            NodeExecution(
                session_id="session_key_test",
                node_name="node_2",
                execution_order=1,
                input_state='{"a": 1, "b": 2, "c": 3, "d": 4}',
                output_state='{"a": 1}',
                started_at="2024-01-15T10:30:01Z",
                duration_ms=10.0,
                status="success",
            ),
        ]
        for exec in execs:
            storage.save_node_execution(exec)

        engine = BlameEngine("session_key_test", storage)
        result = engine.run()

        assert result.blamed_node is not None
        assert result.blamed_node.node_name == "node_2"
        assert result.confidence == "high"

    def test_semantic_blame_uses_pytest_llm(self, storage):
        """Semantic regression via pytest-llm should blame the correct node."""
        session = Session(
            id="session_semantic",
            agent_name="test",
            created_at="2024-01-15T10:30:00Z",
            total_nodes=2,
            status="completed",
        )
        storage.save_session(session)

        baseline = Session(
            id="session_baseline",
            agent_name="test",
            created_at="2024-01-15T10:30:00Z",
            total_nodes=2,
            status="completed",
        )
        storage.save_session(baseline)

        long_output_a = "This is a long output text that exceeds the twenty char threshold"
        long_output_b = "This is a long output text that exceeds the twenty char threshold"
        baseline_text = "This is the expected output text that exceeds the twenty char threshold"

        execs = [
            NodeExecution(
                session_id="session_semantic",
                node_name="node_1",
                execution_order=0,
                input_state='{"query": "hello"}',
                output_state=f'{{"response": "{long_output_a}"}}',
                started_at="2024-01-15T10:30:00Z",
                duration_ms=10.0,
                status="success",
            ),
            NodeExecution(
                session_id="session_semantic",
                node_name="node_2",
                execution_order=1,
                input_state='{"query": "hello"}',
                output_state=f'{{"response": "{long_output_b}"}}',
                started_at="2024-01-15T10:30:01Z",
                duration_ms=10.0,
                status="success",
            ),
        ]
        baseline_execs = [
            NodeExecution(
                session_id="session_baseline",
                node_name="node_1",
                execution_order=0,
                input_state='{"query": "hello"}',
                output_state=f'{{"response": "{baseline_text}"}}',
                started_at="2024-01-15T10:30:00Z",
                duration_ms=10.0,
                status="success",
            ),
            NodeExecution(
                session_id="session_baseline",
                node_name="node_2",
                execution_order=1,
                input_state='{"query": "hello"}',
                output_state=f'{{"response": "{baseline_text}"}}',
                started_at="2024-01-15T10:30:01Z",
                duration_ms=10.0,
                status="success",
            ),
        ]
        for exec in execs:
            storage.save_node_execution(exec)
        for exec in baseline_execs:
            storage.save_node_execution(exec)

        mock_assert = MagicMock(
            side_effect=AssertionError("semantic drift detected for node_2")
        )

        mock_module = MagicMock()
        mock_module.assert_regression = mock_assert

        with patch(
            "langgraph_replay.blame.BlameEngine._try_import_pytest_llm",
            return_value=mock_module,
        ):
            engine = BlameEngine("session_semantic", storage)
            result = engine.run(
                baseline_session_id="session_baseline", use_eval=True
            )

        assert result.blamed_node is not None
        assert result.blamed_node.node_name == "node_2"
        assert result.confidence == "high"
        assert "semantic drift detected" in result.reason

    def test_blame_key_dropped_but_reappears(self, storage):
        """Key that drops temporarily but reappears should NOT be blamed."""
        session = Session(
            id="session_reappear",
            agent_name="test",
            created_at="2024-01-15T10:30:00Z",
            total_nodes=3,
            status="completed",
        )
        storage.save_session(session)

        # Initial: {a, b, c}. Final: {a, b, c} (all present).
        # Node 2 drops c temporarily, but node 3 restores it.
        execs = [
            NodeExecution(
                session_id="session_reappear",
                node_name="node_1",
                execution_order=0,
                input_state='{"a": 1, "b": 2, "c": 3}',
                output_state='{"a": 1, "b": 2, "c": 3}',
                started_at="2024-01-15T10:30:00Z",
                duration_ms=10.0,
                status="success",
            ),
            NodeExecution(
                session_id="session_reappear",
                node_name="node_2",
                execution_order=1,
                input_state='{"a": 1, "b": 2, "c": 3}',
                output_state='{"a": 1, "b": 2}',
                started_at="2024-01-15T10:30:01Z",
                duration_ms=10.0,
                status="success",
            ),
            NodeExecution(
                session_id="session_reappear",
                node_name="node_3",
                execution_order=2,
                input_state='{"a": 1, "b": 2}',
                output_state='{"a": 1, "b": 2, "c": 3}',
                started_at="2024-01-15T10:30:02Z",
                duration_ms=10.0,
                status="success",
            ),
        ]
        for exec in execs:
            storage.save_node_execution(exec)

        engine = BlameEngine("session_reappear", storage)
        result = engine.run()

        assert result.blamed_node is None
        assert result.reason == "No issues found"

    def test_blame_confidence_high_key_never_reappears(self, storage):
        """Key dropped permanently should produce confidence=high."""
        session = Session(
            id="session_perm_drop",
            agent_name="test",
            created_at="2024-01-15T10:30:00Z",
            total_nodes=2,
            status="completed",
        )
        storage.save_session(session)

        execs = [
            NodeExecution(
                session_id="session_perm_drop",
                node_name="node_1",
                execution_order=0,
                input_state='{"a": 1, "b": 2}',
                output_state='{"a": 1, "b": 2}',
                started_at="2024-01-15T10:30:00Z",
                duration_ms=10.0,
                status="success",
            ),
            NodeExecution(
                session_id="session_perm_drop",
                node_name="node_2",
                execution_order=1,
                input_state='{"a": 1, "b": 2}',
                output_state='{"a": 1}',
                started_at="2024-01-15T10:30:01Z",
                duration_ms=10.0,
                status="success",
            ),
        ]
        for exec in execs:
            storage.save_node_execution(exec)

        engine = BlameEngine("session_perm_drop", storage)
        result = engine.run()

        assert result.blamed_node is not None
        assert result.blamed_node.node_name == "node_2"
        assert result.confidence == "high"

    def test_blame_error_takes_priority_over_missing_key(self, storage):
        """Error node should be blamed even if another node dropped a key."""
        session = Session(
            id="session_priority",
            agent_name="test",
            created_at="2024-01-15T10:30:00Z",
            total_nodes=3,
            status="failed",
        )
        storage.save_session(session)

        # node_1 drops key "c" permanently, but node_2 has an error.
        # Error should take priority.
        execs = [
            NodeExecution(
                session_id="session_priority",
                node_name="node_1",
                execution_order=0,
                input_state='{"a": 1, "b": 2, "c": 3}',
                output_state='{"a": 1, "b": 2}',
                started_at="2024-01-15T10:30:00Z",
                duration_ms=10.0,
                status="success",
            ),
            NodeExecution(
                session_id="session_priority",
                node_name="node_2",
                execution_order=1,
                input_state='{"a": 1, "b": 2}',
                output_state="{}",
                started_at="2024-01-15T10:30:01Z",
                duration_ms=5.0,
                status="error",
                error_message="RuntimeError: something broke",
            ),
            NodeExecution(
                session_id="session_priority",
                node_name="node_3",
                execution_order=2,
                input_state='{}',
                output_state='{"a": 1}',
                started_at="2024-01-15T10:30:02Z",
                duration_ms=10.0,
                status="success",
            ),
        ]
        for exec in execs:
            storage.save_node_execution(exec)

        engine = BlameEngine("session_priority", storage)
        result = engine.run()

        assert result.blamed_node is not None
        assert result.blamed_node.node_name == "node_2"
        assert result.confidence == "high"

    def test_auto_baseline_selection(self, storage):
        """Auto-baseline finds most recent completed session for same agent."""
        # First session: completed, 3 nodes, all success
        session_a = Session(
            id="session_auto_a",
            agent_name="auto_agent",
            created_at="2024-01-15T10:00:00Z",
            total_nodes=3,
            status="completed",
        )
        storage.save_session(session_a)
        for i in range(3):
            storage.save_node_execution(NodeExecution(
                session_id="session_auto_a",
                node_name=f"node_{i}",
                execution_order=i,
                input_state=f'{{"step": {i}}}',
                output_state=f'{{"step": {i + 1}}}',
                started_at="2024-01-15T10:00:00Z",
                duration_ms=10.0,
                status="success",
            ))

        # Second session: completed, 3 nodes, one node has degraded output
        session_b = Session(
            id="session_auto_b",
            agent_name="auto_agent",
            created_at="2024-01-15T11:00:00Z",
            total_nodes=3,
            status="completed",
        )
        storage.save_session(session_b)
        for i in range(3):
            storage.save_node_execution(NodeExecution(
                session_id="session_auto_b",
                node_name=f"node_{i}",
                execution_order=i,
                input_state=f'{{"step": {i}}}',
                output_state=f'{{"step": {i + 1}}}' if i != 2 else '{}',
                started_at="2024-01-15T11:00:00Z",
                duration_ms=10.0,
                status="success",
            ))

        # Run blame on session_b with no baseline — should auto-select session_a
        engine = BlameEngine("session_auto_b", storage)
        result = engine.run(use_eval=False)

        # Should run without error; structural blame finds dropped key
        assert result.blamed_node is not None
        assert result.blamed_node.node_name == "node_2"

    def test_no_baseline_available(self, storage):
        """Graceful fallback when no baseline session exists for the agent."""
        session = Session(
            id="session_no_base",
            agent_name="unique_agent_xyz",
            created_at="2024-01-15T10:00:00Z",
            total_nodes=2,
            status="completed",
        )
        storage.save_session(session)
        for i in range(2):
            storage.save_node_execution(NodeExecution(
                session_id="session_no_base",
                node_name=f"node_{i}",
                execution_order=i,
                input_state=f'{{"step": {i}}}',
                output_state=f'{{"step": {i + 1}}}',
                started_at="2024-01-15T10:00:00Z",
                duration_ms=10.0,
                status="success",
            ))

        # Run with use_eval=True but no baseline — should fall back gracefully
        engine = BlameEngine("session_no_base", storage)
        result = engine.run(use_eval=True)

        # No structural issues, so blamed_node should be None
        assert result.blamed_node is None
        assert result.reason == "No issues found"
