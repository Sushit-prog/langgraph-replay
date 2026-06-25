"""Tests for diff functions."""

import pytest

from langgraph_replay.diff import compute_state_diff, compute_session_diff, StateDiff
from langgraph_replay.storage import NodeExecution


class TestStateDiff:
    """Tests for compute_state_diff."""

    def test_compute_state_diff_added_keys(self):
        """Keys in after but not in before should be in added."""
        before = {"a": 1}
        after = {"a": 1, "b": 2}
        diff = compute_state_diff(before, after)
        assert diff.added == {"b": 2}
        assert diff.removed == {}
        assert diff.modified == {}

    def test_compute_state_diff_removed_keys(self):
        """Keys in before but not in after should be in removed."""
        before = {"a": 1, "b": 2}
        after = {"a": 1}
        diff = compute_state_diff(before, after)
        assert diff.removed == {"b": 2}
        assert diff.added == {}
        assert diff.modified == {}

    def test_compute_state_diff_modified_keys(self):
        """Keys in both but different values should be in modified."""
        before = {"a": 1}
        after = {"a": 2}
        diff = compute_state_diff(before, after)
        assert diff.modified == {"a": {"before": 1, "after": 2}}
        assert diff.added == {}
        assert diff.removed == {}

    def test_compute_state_diff_unchanged_keys(self):
        """Keys with same value should be in unchanged."""
        before = {"a": 1, "b": 2}
        after = {"a": 1, "b": 2}
        diff = compute_state_diff(before, after)
        assert set(diff.unchanged) == {"a", "b"}
        assert diff.added == {}
        assert diff.removed == {}
        assert diff.modified == {}

    def test_compute_state_diff_empty_states(self):
        """Both empty should produce empty diff."""
        diff = compute_state_diff({}, {})
        assert diff.added == {}
        assert diff.removed == {}
        assert diff.modified == {}
        assert diff.unchanged == []

    def test_compute_state_diff_mixed_changes(self):
        """Mix of added, removed, modified, unchanged."""
        before = {"a": 1, "b": 2, "c": 3}
        after = {"a": 1, "b": 99, "d": 4}
        diff = compute_state_diff(before, after)
        assert diff.added == {"d": 4}
        assert diff.removed == {"c": 3}
        assert diff.modified == {"b": {"before": 2, "after": 99}}
        assert diff.unchanged == ["a"]


class TestSessionDiff:
    """Tests for compute_session_diff."""

    def test_session_diff_nodes_only_in_a(self):
        """Nodes in A but not B should appear in nodes_only_in_a."""
        execs_a = [
            NodeExecution(
                session_id="a",
                node_name="node_x",
                execution_order=0,
                status="success",
                duration_ms=10.0,
            ),
            NodeExecution(
                session_id="a",
                node_name="node_y",
                execution_order=1,
                status="success",
                duration_ms=10.0,
            ),
        ]
        execs_b = [
            NodeExecution(
                session_id="b",
                node_name="node_x",
                execution_order=0,
                status="success",
                duration_ms=10.0,
            ),
        ]
        diff = compute_session_diff(execs_a, execs_b)
        assert "node_y" in diff.nodes_only_in_a
        assert diff.nodes_only_in_b == []

    def test_session_diff_nodes_only_in_b(self):
        """Nodes in B but not A should appear in nodes_only_in_b."""
        execs_a = [
            NodeExecution(
                session_id="a",
                node_name="node_x",
                execution_order=0,
                status="success",
                duration_ms=10.0,
            ),
        ]
        execs_b = [
            NodeExecution(
                session_id="b",
                node_name="node_x",
                execution_order=0,
                status="success",
                duration_ms=10.0,
            ),
            NodeExecution(
                session_id="b",
                node_name="node_z",
                execution_order=1,
                status="success",
                duration_ms=10.0,
            ),
        ]
        diff = compute_session_diff(execs_a, execs_b)
        assert "node_z" in diff.nodes_only_in_b
        assert diff.nodes_only_in_a == []

    def test_session_diff_duration_comparison(self):
        """Duration diff should be b - a (negative means b was faster)."""
        execs_a = [
            NodeExecution(
                session_id="a",
                node_name="node_x",
                execution_order=0,
                status="success",
                duration_ms=100.0,
                output_state='{"key": "val"}',
            ),
        ]
        execs_b = [
            NodeExecution(
                session_id="b",
                node_name="node_x",
                execution_order=0,
                status="success",
                duration_ms=50.0,
                output_state='{"key": "val"}',
            ),
        ]
        diff = compute_session_diff(execs_a, execs_b)
        assert len(diff.nodes_in_both) == 1
        assert diff.nodes_in_both[0].duration_diff_ms == -50.0

    def test_session_diff_status_changed(self):
        """Status difference should be flagged."""
        execs_a = [
            NodeExecution(
                session_id="a",
                node_name="node_x",
                execution_order=0,
                status="success",
                duration_ms=10.0,
            ),
        ]
        execs_b = [
            NodeExecution(
                session_id="b",
                node_name="node_x",
                execution_order=0,
                status="error",
                duration_ms=10.0,
            ),
        ]
        diff = compute_session_diff(execs_a, execs_b)
        assert diff.nodes_in_both[0].status_changed is True

    def test_session_diff_state_changes(self):
        """State differences should be captured."""
        execs_a = [
            NodeExecution(
                session_id="a",
                node_name="node_x",
                execution_order=0,
                status="success",
                duration_ms=10.0,
                output_state='{"a": 1}',
            ),
        ]
        execs_b = [
            NodeExecution(
                session_id="b",
                node_name="node_x",
                execution_order=0,
                status="success",
                duration_ms=10.0,
                output_state='{"a": 2}',
            ),
        ]
        diff = compute_session_diff(execs_a, execs_b)
        assert diff.nodes_in_both[0].state_diff.modified == {"a": {"before": 1, "after": 2}}
