"""LangGraph callback handler that records node executions."""

import copy
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from langchain_core.callbacks.base import BaseCallbackHandler

from langgraph_replay.storage import NodeExecution, ReplayStorage, Session, _serialize_state

logger = logging.getLogger(__name__)


class LangGraphRecorder(BaseCallbackHandler):
    """Callback handler that records every node execution in a LangGraph run.

    Usage::

        recorder = LangGraphRecorder(session_name="my_agent")
        result = graph.invoke(
            initial_state,
            config={"callbacks": [recorder]}
        )
        session_id = recorder.session_id
        print(f"Recorded: {session_id}")
    """

    def __init__(
        self,
        session_name: str = "unnamed",
        storage: Optional[ReplayStorage] = None,
        metadata: Optional[dict] = None,
    ):
        """Initialize the recorder.

        Args:
            session_name: Human-readable name for this run.
            storage: ReplayStorage instance. Creates default if None.
            metadata: Arbitrary dict stored with the session.
        """
        super().__init__()
        self._session_name = session_name
        self._storage = storage or ReplayStorage()
        self._metadata = metadata or {}
        self._session_id = f"session_{uuid4().hex[:8]}"
        self._node_stack: list[dict] = []
        self._execution_order: int = 0
        self._node_executions: list[NodeExecution] = []
        self._start_time: Optional[float] = None

    @property
    def session_id(self) -> str:
        """Returns the session ID for this recording."""
        return self._session_id

    def on_chain_start(
        self,
        serialized: dict,
        inputs: dict,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain/node starts.

        Only records if this looks like a LangGraph node:
        - name is in kwargs
        - parent_run_id is not None (top-level graph call has no parent)
        """
        node_name = kwargs.get("name", "")
        if not node_name or parent_run_id is None:
            return

        self._node_stack.append(
            {
                "node_name": node_name,
                "input_state": copy.deepcopy(inputs),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "start_time": time.perf_counter(),
            }
        )
        if self._start_time is None:
            self._start_time = time.perf_counter()

    def on_chain_end(
        self, outputs: Any, *, run_id: Any, **kwargs: Any
    ) -> None:
        """Called when a chain/node completes successfully."""
        if not self._node_stack:
            return

        ctx = self._node_stack.pop()
        duration_ms = (time.perf_counter() - ctx["start_time"]) * 1000

        execution = NodeExecution(
            session_id=self._session_id,
            node_name=ctx["node_name"],
            execution_order=self._execution_order,
            input_state=_serialize_state(ctx["input_state"]),
            output_state=_serialize_state(outputs),
            started_at=ctx["started_at"],
            duration_ms=round(duration_ms, 2),
            status="success",
            llm_calls=0,
        )
        self._execution_order += 1
        self._node_executions.append(execution)

        try:
            self._storage.save_node_execution(execution)
        except Exception as e:
            logger.warning(f"Failed to save node execution: {e}")

    def on_chain_error(
        self, error: BaseException, *, run_id: Any, **kwargs: Any
    ) -> None:
        """Called when a chain/node raises an exception."""
        if not self._node_stack:
            return

        ctx = self._node_stack.pop()
        duration_ms = (time.perf_counter() - ctx["start_time"]) * 1000

        execution = NodeExecution(
            session_id=self._session_id,
            node_name=ctx["node_name"],
            execution_order=self._execution_order,
            input_state=_serialize_state(ctx["input_state"]),
            output_state="{}",
            started_at=ctx["started_at"],
            duration_ms=round(duration_ms, 2),
            status="error",
            error_message=str(error),
            llm_calls=0,
        )
        self._execution_order += 1
        self._node_executions.append(execution)

        try:
            self._storage.save_node_execution(execution)
        except Exception as e:
            logger.warning(f"Failed to save node execution: {e}")

    def finalize(
        self, final_output: Any = None, status: str = "completed"
    ) -> str:
        """Save the session record to storage.

        Call this after graph.invoke() completes.

        Args:
            final_output: The final output from the graph.
            status: Session status ("completed", "failed", "interrupted").

        Returns:
            The session ID.
        """
        session = Session(
            id=self._session_id,
            agent_name=self._session_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            total_nodes=len(self._node_executions),
            status=status,
            final_output=_serialize_state(final_output),
            metadata=self._metadata,
        )
        try:
            self._storage.save_session(session)
        except Exception as e:
            logger.warning(f"Failed to save session: {e}")
        return self._session_id


class record_session:
    """Context manager that wraps a LangGraph invocation.

    Usage::

        with record_session("research_agent") as rec:
            result = graph.invoke(state, config={"callbacks": [rec]})
        print(rec.session_id)
    """

    def __init__(
        self,
        name: str,
        storage: Optional[ReplayStorage] = None,
        metadata: Optional[dict] = None,
    ):
        """Initialize the context manager.

        Args:
            name: Human-readable session name.
            storage: Optional ReplayStorage instance.
            metadata: Optional metadata dict.
        """
        self._recorder = LangGraphRecorder(
            session_name=name, storage=storage, metadata=metadata
        )

    def __enter__(self) -> LangGraphRecorder:
        """Enter the context, returning the recorder."""
        return self._recorder

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context, finalizing the session.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        status = "failed" if exc_type is not None else "completed"
        self._recorder.finalize(status=status)
