"""Shared test fixtures for langgraph-replay."""

import pytest
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END

from langgraph_replay.storage import ReplayStorage, Session, NodeExecution


class AgentState(TypedDict):
    messages: List[str]
    step: int


@pytest.fixture
def storage(tmp_path):
    """ReplayStorage backed by a temp SQLite file."""
    return ReplayStorage(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def sample_session():
    """Returns a Session object with realistic test data."""
    return Session(
        id="session_test01",
        agent_name="test_agent",
        created_at="2024-01-15T10:30:00Z",
        total_nodes=3,
        status="completed",
        final_output='{"messages": ["hello", "world"], "step": 3}',
        metadata={"test": True},
    )


@pytest.fixture
def sample_executions():
    """Returns a list of 5 NodeExecution objects simulating a 5-node run.
    Node 3 has status=error."""
    return [
        NodeExecution(
            session_id="session_test01",
            node_name="node_1",
            execution_order=0,
            input_state='{"messages": [], "step": 0}',
            output_state='{"messages": ["started"], "step": 1}',
            started_at="2024-01-15T10:30:00Z",
            duration_ms=10.5,
            status="success",
        ),
        NodeExecution(
            session_id="session_test01",
            node_name="node_2",
            execution_order=1,
            input_state='{"messages": ["started"], "step": 1}',
            output_state='{"messages": ["started", "processing"], "step": 2}',
            started_at="2024-01-15T10:30:00Z",
            duration_ms=15.2,
            status="success",
        ),
        NodeExecution(
            session_id="session_test01",
            node_name="node_3",
            execution_order=2,
            input_state='{"messages": ["started", "processing"], "step": 2}',
            output_state="{}",
            started_at="2024-01-15T10:30:00Z",
            duration_ms=5.0,
            status="error",
            error_message="ValueError: processing failed",
        ),
        NodeExecution(
            session_id="session_test01",
            node_name="node_4",
            execution_order=3,
            input_state='{"messages": ["started", "processing"], "step": 2}',
            output_state='{"messages": ["started", "processing", "recovered"], "step": 3}',
            started_at="2024-01-15T10:30:00Z",
            duration_ms=8.3,
            status="success",
        ),
        NodeExecution(
            session_id="session_test01",
            node_name="node_5",
            execution_order=4,
            input_state='{"messages": ["started", "processing", "recovered"], "step": 3}',
            output_state='{"messages": ["started", "processing", "recovered", "done"], "step": 4}',
            started_at="2024-01-15T10:30:00Z",
            duration_ms=12.1,
            status="success",
        ),
    ]


@pytest.fixture
def mock_graph():
    """A minimal LangGraph graph for testing the recorder.
    Three nodes: node_a -> node_b -> node_c
    State: {"messages": list, "step": int}
    Each node appends to messages and increments step.
    """

    def node_a(state: AgentState) -> dict:
        return {
            "messages": state["messages"] + ["node_a executed"],
            "step": state["step"] + 1,
        }

    def node_b(state: AgentState) -> dict:
        return {
            "messages": state["messages"] + ["node_b executed"],
            "step": state["step"] + 1,
        }

    def node_c(state: AgentState) -> dict:
        return {
            "messages": state["messages"] + ["node_c executed"],
            "step": state["step"] + 1,
        }

    graph = StateGraph(AgentState)
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)
    graph.add_node("node_c", node_c)
    graph.set_entry_point("node_a")
    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", "node_c")
    graph.add_edge("node_c", END)

    return graph.compile()


@pytest.fixture
def error_graph():
    """A LangGraph graph where node_b raises ValueError."""

    def node_a(state: AgentState) -> dict:
        return {
            "messages": state["messages"] + ["node_a executed"],
            "step": state["step"] + 1,
        }

    def node_b(state: AgentState) -> dict:
        raise ValueError("node_b failed intentionally")

    def node_c(state: AgentState) -> dict:
        return {
            "messages": state["messages"] + ["node_c executed"],
            "step": state["step"] + 1,
        }

    graph = StateGraph(AgentState)
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)
    graph.add_node("node_c", node_c)
    graph.set_entry_point("node_a")
    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", "node_c")
    graph.add_edge("node_c", END)

    return graph.compile()
