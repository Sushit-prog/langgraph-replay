"""Tests for ReplayStorage."""

import pytest
import json

from langgraph_replay.storage import ReplayStorage, Session, NodeExecution


class TestReplayStorage:
    """Tests for the SQLite storage layer."""

    def test_save_and_retrieve_session(self, storage, sample_session):
        """Save a session and retrieve it."""
        storage.save_session(sample_session)
        retrieved = storage.get_session(sample_session.id)
        assert retrieved is not None
        assert retrieved.id == sample_session.id
        assert retrieved.agent_name == sample_session.agent_name

    def test_save_and_retrieve_node_executions(self, storage, sample_executions):
        """Save node executions and retrieve them."""
        for exec in sample_executions:
            storage.save_node_execution(exec)

        retrieved = storage.get_node_executions("session_test01")
        assert len(retrieved) == 5
        assert retrieved[0].node_name == "node_1"
        assert retrieved[4].node_name == "node_5"

    def test_list_sessions_ordered_by_recency(self, storage):
        """List sessions and verify they are ordered newest first."""
        for i in range(5):
            session = Session(
                id=f"session_{i:02d}",
                agent_name="test",
                created_at=f"2024-01-15T10:{30+i}:00Z",
                total_nodes=i,
                status="completed",
            )
            storage.save_session(session)

        sessions = storage.list_sessions(limit=3)
        assert len(sessions) == 3
        # Newest first
        assert sessions[0].id == "session_04"
        assert sessions[2].id == "session_02"

    def test_delete_session_removes_executions(self, storage, sample_session, sample_executions):
        """Delete session and verify executions are also removed."""
        storage.save_session(sample_session)
        for exec in sample_executions:
            storage.save_node_execution(exec)

        deleted = storage.delete_session("session_test01")
        assert deleted is True

        assert storage.get_session("session_test01") is None
        assert len(storage.get_node_executions("session_test01")) == 0

    def test_get_nonexistent_session_returns_none(self, storage):
        """Getting a non-existent session returns None."""
        result = storage.get_session("nonexistent")
        assert result is None

    def test_search_by_status(self, storage):
        """Search sessions filtered by status."""
        for i, status in enumerate(["completed", "failed", "completed"]):
            session = Session(
                id=f"session_status_{i}",
                agent_name="test",
                created_at="2024-01-15T10:00:00Z",
                total_nodes=1,
                status=status,
            )
            storage.save_session(session)

        completed = storage.search_sessions(status="completed")
        assert len(completed) == 2

        failed = storage.search_sessions(status="failed")
        assert len(failed) == 1

    def test_search_by_agent_name(self, storage):
        """Search sessions filtered by agent name."""
        for i, name in enumerate(["agent_a", "agent_b", "agent_a"]):
            session = Session(
                id=f"session_agent_{i}",
                agent_name=name,
                created_at="2024-01-15T10:00:00Z",
                total_nodes=1,
                status="completed",
            )
            storage.save_session(session)

        agent_a = storage.search_sessions(agent_name="agent_a")
        assert len(agent_a) == 2

        agent_b = storage.search_sessions(agent_name="agent_b")
        assert len(agent_b) == 1
