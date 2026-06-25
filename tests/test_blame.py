"""Tests for the BlameEngine."""

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

    def test_blame_returns_none_when_no_issues(self, storage):
        """All nodes succeed with no state drops returns None."""
        session = Session(
            id="session_clean",
            agent_name="test",
            created_at="2024-01-15T10:30:00Z",
            total_nodes=2,
            status="completed",
        )
        storage.save_session(session)

        execs = [
            NodeExecution(
                session_id="session_clean",
                node_name="node_1",
                execution_order=0,
                input_state='{"a": 1}',
                output_state='{"a": 1, "b": 2}',
                started_at="2024-01-15T10:30:00Z",
                duration_ms=10.0,
                status="success",
            ),
            NodeExecution(
                session_id="session_clean",
                node_name="node_2",
                execution_order=1,
                input_state='{"a": 1, "b": 2}',
                output_state='{"a": 1, "b": 2, "c": 3}',
                started_at="2024-01-15T10:30:01Z",
                duration_ms=10.0,
                status="success",
            ),
        ]
        for exec in execs:
            storage.save_node_execution(exec)

        engine = BlameEngine("session_clean", storage)
        result = engine.run()

        assert result.blamed_node is None
        assert result.confidence == "low"
        assert result.reason == "No issues found"

    def test_blame_confidence_high_on_error(self, storage, sample_executions):
        """Error node should produce confidence=high."""
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

        assert result.confidence == "high"
        assert "error" in result.reason.lower()

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
