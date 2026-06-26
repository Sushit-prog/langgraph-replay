"""Tests for the LangGraphRecorder."""

import pytest

from langgraph_replay.recorder import LangGraphRecorder, record_session, arecord_session


class TestLangGraphRecorder:
    """Tests for LangGraphRecorder against mock_graph fixture."""

    def test_recorder_captures_all_nodes(self, mock_graph, storage):
        """Run mock_graph with recorder and assert 3 nodes captured."""
        recorder = LangGraphRecorder(session_name="test", storage=storage)
        result = mock_graph.invoke(
            {"messages": [], "step": 0},
            config={"callbacks": [recorder]},
        )
        assert len(recorder._node_executions) == 3

    def test_recorder_captures_state_transitions(self, mock_graph, storage):
        """Assert each NodeExecution has valid state and step increments."""
        recorder = LangGraphRecorder(session_name="test", storage=storage)
        mock_graph.invoke(
            {"messages": [], "step": 0},
            config={"callbacks": [recorder]},
        )

        for i, exec in enumerate(recorder._node_executions):
            assert exec.input_state is not None
            assert exec.output_state is not None
            input_state = __import__("json").loads(exec.input_state)
            output_state = __import__("json").loads(exec.output_state)
            assert output_state["step"] == input_state["step"] + 1

    def test_recorder_computes_duration(self, mock_graph, storage):
        """Assert each NodeExecution has duration_ms > 0."""
        recorder = LangGraphRecorder(session_name="test", storage=storage)
        mock_graph.invoke(
            {"messages": [], "step": 0},
            config={"callbacks": [recorder]},
        )

        for exec in recorder._node_executions:
            assert exec.duration_ms > 0

    def test_recorder_saves_to_storage(self, mock_graph, storage):
        """Run mock_graph, finalize, and verify storage has session and executions."""
        recorder = LangGraphRecorder(session_name="test", storage=storage)
        mock_graph.invoke(
            {"messages": [], "step": 0},
            config={"callbacks": [recorder]},
        )
        recorder.finalize()

        session = storage.get_session(recorder.session_id)
        assert session is not None
        assert session.agent_name == "test"
        assert session.total_nodes == 3

        executions = storage.get_node_executions(recorder.session_id)
        assert len(executions) == 3

    def test_context_manager(self, mock_graph, storage):
        """Use record_session context manager and verify session is saved."""
        with record_session("test_ctx", storage=storage) as rec:
            mock_graph.invoke(
                {"messages": [], "step": 0},
                config={"callbacks": [rec]},
            )

        session = storage.get_session(rec.session_id)
        assert session is not None
        assert session.status == "completed"
        assert session.total_nodes == 3

    def test_recorder_handles_error_node(self, error_graph, storage):
        """Build graph where node_b raises, verify error is recorded."""
        recorder = LangGraphRecorder(session_name="test_error", storage=storage)

        with pytest.raises(ValueError, match="node_b failed"):
            error_graph.invoke(
                {"messages": [], "step": 0},
                config={"callbacks": [recorder]},
            )

        recorder.finalize()

        executions = storage.get_node_executions(recorder.session_id)
        error_exec = next(e for e in executions if e.node_name == "node_b")
        assert error_exec.status == "error"
        assert "node_b failed" in error_exec.error_message


class TestAreordSession:
    """Tests for the async arecord_session context manager."""

    @pytest.mark.asyncio
    async def test_arecord_session_context_manager(self, storage):
        """Use arecord_session as async context manager and verify session is saved."""
        async with arecord_session("test_async", storage=storage) as rec:
            # Simulate an async agent by manually firing callbacks
            from uuid import uuid4
            from datetime import datetime, timezone
            import time

            run_id = uuid4()
            parent_run_id = uuid4()

            rec.on_chain_start(
                {},
                {"messages": [], "step": 0},
                run_id=run_id,
                parent_run_id=parent_run_id,
                name="node_a",
            )
            rec.on_chain_end(
                {"messages": ["a"], "step": 1},
                run_id=run_id,
            )

        # Session should be saved after context exits
        session = storage.get_session(rec.session_id)
        assert session is not None
        assert session.status == "completed"
        assert session.agent_name == "test_async"
        assert len(rec._node_executions) == 1
        assert rec._node_executions[0].node_name == "node_a"

    @pytest.mark.asyncio
    async def test_arecord_session_handles_exception(self, storage):
        """arecord_session should mark session as failed on exception."""
        with pytest.raises(RuntimeError, match="async boom"):
            async with arecord_session("test_async_fail", storage=storage) as rec:
                raise RuntimeError("async boom")

        session = storage.get_session(rec.session_id)
        assert session is not None
        assert session.status == "failed"
