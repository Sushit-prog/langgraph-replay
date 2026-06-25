"""Replay engine for navigating through recorded LangGraph sessions."""

from typing import Optional

from langgraph_replay.storage import NodeExecution, ReplayStorage, _deserialize_state


class ReplayEngine:
    """Step-by-step navigation through a recorded LangGraph session.

    Usage::

        engine = ReplayEngine("session_abc123")
        while not engine.is_at_end():
            exec = engine.step_forward()
            print(f"Node: {exec.node_name}, Duration: {exec.duration_ms}ms")
    """

    def __init__(self, session_id: str, storage: Optional[ReplayStorage] = None):
        """Initialize the replay engine.

        Args:
            session_id: The session ID to replay.
            storage: Optional ReplayStorage instance.
        """
        self._storage = storage or ReplayStorage()
        self.session = self._storage.get_session(session_id)
        self.executions = self._storage.get_node_executions(session_id)
        self._current_index = 0

    def step_forward(self) -> Optional[NodeExecution]:
        """Move to the next node.

        Returns:
            NodeExecution at the new position, or None if at end.
        """
        if self._current_index >= len(self.executions):
            return None
        exec = self.executions[self._current_index]
        self._current_index += 1
        return exec

    def step_backward(self) -> Optional[NodeExecution]:
        """Move to the previous node.

        Returns:
            NodeExecution at the new position, or None if at start.
        """
        if self._current_index <= 0:
            return None
        self._current_index -= 1
        return self.executions[self._current_index]

    def jump_to(self, index: int) -> NodeExecution:
        """Jump to a specific node by execution_order index.

        Args:
            index: The execution order index (0-based).

        Returns:
            NodeExecution at the specified index.

        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index >= len(self.executions):
            raise IndexError(f"Index {index} out of range (0-{len(self.executions)-1})")
        self._current_index = index
        return self.executions[index]

    def current(self) -> Optional[NodeExecution]:
        """Returns the current NodeExecution.

        Returns:
            Current NodeExecution or None if no executions.
        """
        if not self.executions or self._current_index >= len(self.executions):
            return None
        return self.executions[self._current_index]

    def state_at(self, index: int) -> dict:
        """Returns the input state at a given node index.

        Args:
            index: The execution order index.

        Returns:
            Dictionary of the input state at that node.
        """
        if index < 0 or index >= len(self.executions):
            return {}
        return _deserialize_state(self.executions[index].input_state)

    def is_at_end(self) -> bool:
        """Check if we're at the end of the execution list."""
        return self._current_index >= len(self.executions)

    def is_at_start(self) -> bool:
        """Check if we're at the start of the execution list."""
        return self._current_index <= 0

    @property
    def total_nodes(self) -> int:
        """Returns the total number of nodes in this session."""
        return len(self.executions)
