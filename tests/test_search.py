"""Tests for semantic session search."""

import pytest

from langgraph_replay.storage import Session, NodeExecution, ReplayStorage
from langgraph_replay.search import SessionSearchEngine, SearchResult


class TestSessionSearchEngine:
    """Tests for SessionSearchEngine."""

    def test_session_to_text_format(self, storage):
        """_session_to_text returns formatted text."""
        session = Session(
            id="s_text",
            agent_name="research_bot",
            created_at="2024-01-15T10:00:00Z",
            total_nodes=3,
            status="completed",
            final_output='{"answer": "done"}',
        )
        storage.save_session(session)
        for i, name in enumerate(["fetch", "summarize", "format"]):
            storage.save_node_execution(NodeExecution(
                session_id="s_text",
                node_name=name,
                execution_order=i,
                input_state='{"ctx": "hello"}',
                output_state='{"ctx": "hello", "step": %d}' % i,
                started_at="2024-01-15T10:00:00Z",
                duration_ms=10.0,
                status="success",
            ))

        engine = SessionSearchEngine(storage)
        text = engine._session_to_text("s_text")

        assert "research_bot" in text
        assert "fetch" in text
        assert "summarize" in text
        assert "format" in text
        assert "Status: completed" in text
        assert "ctx" in text

    def test_search_returns_results(self, storage):
        """Search finds matching sessions."""
        for name in ["alpha_agent", "beta_agent", "gamma_agent"]:
            s = Session(
                id=f"s_{name}",
                agent_name=name,
                created_at="2024-01-15T10:00:00Z",
                total_nodes=1,
                status="completed",
            )
            storage.save_session(s)
            storage.save_node_execution(NodeExecution(
                session_id=f"s_{name}",
                node_name="node_1",
                execution_order=0,
                input_state='{}',
                output_state='{}',
                started_at="2024-01-15T10:00:00Z",
                duration_ms=10.0,
                status="success",
            ))

        engine = SessionSearchEngine(storage)
        results = engine.search("alpha_agent", threshold=0.1)
        assert len(results) > 0
        assert any(r.agent_name == "alpha_agent" for r in results)

    def test_search_threshold_filters(self, storage):
        """Higher threshold returns fewer results."""
        for name in ["agent_a", "agent_b"]:
            s = Session(
                id=f"s_{name}",
                agent_name=name,
                created_at="2024-01-15T10:00:00Z",
                total_nodes=1,
                status="completed",
            )
            storage.save_session(s)
            storage.save_node_execution(NodeExecution(
                session_id=f"s_{name}",
                node_name="node_1",
                execution_order=0,
                input_state='{}',
                output_state='{}',
                started_at="2024-01-15T10:00:00Z",
                duration_ms=10.0,
                status="success",
            ))

        engine = SessionSearchEngine(storage)
        low = engine.search("agent", threshold=0.1)
        high = engine.search("agent", threshold=0.99)
        assert len(low) >= len(high)

    def test_find_similar_excludes_self(self, storage):
        """find_similar returns other sessions, not the query itself as top."""
        for name in ["agent_x", "agent_y"]:
            s = Session(
                id=f"s_{name}",
                agent_name=name,
                created_at="2024-01-15T10:00:00Z",
                total_nodes=1,
                status="completed",
            )
            storage.save_session(s)
            storage.save_node_execution(NodeExecution(
                session_id=f"s_{name}",
                node_name="node_1",
                execution_order=0,
                input_state='{}',
                output_state='{}',
                started_at="2024-01-15T10:00:00Z",
                duration_ms=10.0,
                status="success",
            ))

        engine = SessionSearchEngine(storage)
        results = engine.find_similar("s_agent_x")
        session_ids = [r.session_id for r in results]
        assert len(results) >= 1

    def test_search_empty_storage(self):
        """Search on empty storage returns empty list."""
        import tempfile, os
        db = os.path.join(tempfile.mkdtemp(), "empty.db")
        storage = ReplayStorage(db_path=db)
        engine = SessionSearchEngine(storage)
        results = engine.search("anything")
        assert results == []
        storage.close()

    def test_search_result_ordered_by_score(self, storage):
        """Results are ordered by score descending."""
        for i in range(3):
            s = Session(
                id=f"s_{i}",
                agent_name=f"agent_{i}",
                created_at=f"2024-01-15T10:00:0{i}Z",
                total_nodes=1,
                status="completed",
            )
            storage.save_session(s)
            storage.save_node_execution(NodeExecution(
                session_id=f"s_{i}",
                node_name="node_1",
                execution_order=0,
                input_state='{}',
                output_state='{}',
                started_at="2024-01-15T10:00:00Z",
                duration_ms=10.0,
                status="success",
            ))

        engine = SessionSearchEngine(storage)
        results = engine.search("agent", threshold=0.1)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score