"""Example: Simple 3-node LangGraph agent with replay recording.

This agent processes text through three steps:
1. Summarize: Condenses input text
2. Extract: Pulls out key points
3. Format: Structures the final output

Uses a mock LLM (no real API calls needed).
"""

from typing import TypedDict, List, Annotated
from unittest.mock import patch, MagicMock

from langgraph.graph import StateGraph, END
from langgraph_replay import LangGraphRecorder, record_session, BlameEngine, ReplayStorage


# Define the state schema
class AgentState(TypedDict):
    text: str
    summary: str
    key_points: List[str]
    formatted_output: str


# Node implementations (these would normally call an LLM)
def summarize_node(state: AgentState) -> dict:
    """Summarize the input text."""
    text = state["text"]
    # In real usage, this would be: llm.invoke(f"Summarize: {text}")
    summary = f"Summary of: {text[:50]}..."
    return {"summary": summary}


def extract_node(state: AgentState) -> dict:
    """Extract key points from the summary."""
    summary = state["summary"]
    # In real usage, this would be: llm.invoke(f"Extract key points: {summary}")
    key_points = [
        "Key point 1 from the text",
        "Key point 2 from the text",
        "Key point 3 from the text",
    ]
    return {"key_points": key_points}


def format_node(state: AgentState) -> dict:
    """Format the output nicely."""
    summary = state["summary"]
    key_points = state["key_points"]
    formatted = f"## Analysis\n\n{summary}\n\n### Key Points\n"
    for point in key_points:
        formatted += f"- {point}\n"
    return {"formatted_output": formatted}


def build_graph():
    """Build the 3-node agent graph."""
    graph = StateGraph(AgentState)
    graph.add_node("summarize", summarize_node)
    graph.add_node("extract", extract_node)
    graph.add_node("format", format_node)
    graph.set_entry_point("summarize")
    graph.add_edge("summarize", "extract")
    graph.add_edge("extract", "format")
    graph.add_edge("format", END)
    return graph.compile()


def example_recorder():
    """Example 1: Using LangGraphRecorder directly."""
    print("=" * 60)
    print("Example 1: LangGraphRecorder")
    print("=" * 60)

    graph = build_graph()
    storage = ReplayStorage()

    # Create recorder with session name
    recorder = LangGraphRecorder(
        session_name="simple_agent_example",
        storage=storage,
        metadata={"example": "recorder_direct"},
    )

    # Run the graph with the recorder as callback
    result = graph.invoke(
        {"text": "LangGraph is a framework for building stateful, multi-actor applications with LLMs.", "summary": "", "key_points": [], "formatted_output": ""},
        config={"callbacks": [recorder]},
    )

    # Finalize the session (saves to storage)
    session_id = recorder.finalize(final_output=result)
    print(f"Recorded session: {session_id}")
    print(f"Final output:\n{result['formatted_output']}")

    # Now we can examine what was recorded
    engine = ReplayStorage()
    executions = engine.get_node_executions(session_id)
    print(f"\nRecorded {len(executions)} node executions:")
    for exec in executions:
        print(f"  [{exec.execution_order}] {exec.node_name}: {exec.duration_ms:.1f}ms ({exec.status})")


def example_context_manager():
    """Example 2: Using the record_session context manager."""
    print("\n" + "=" * 60)
    print("Example 2: record_session context manager")
    print("=" * 60)

    graph = build_graph()

    # The context manager handles finalize automatically
    with record_session("context_manager_example") as rec:
        result = graph.invoke(
            {"text": "Context managers make recording easier!", "summary": "", "key_points": [], "formatted_output": ""},
            config={"callbacks": [rec]},
        )

    print(f"Recorded session: {rec.session_id}")
    print(f"Nodes recorded: {len(rec._node_executions)}")


def example_blame():
    """Example 3: Using blame analysis to find the failing node."""
    print("\n" + "=" * 60)
    print("Example 3: Blame analysis")
    print("=" * 60)

    graph = build_graph()

    # Record a successful run
    with record_session("blame_example") as rec:
        result = graph.invoke(
            {"text": "Test blame analysis", "summary": "", "key_points": [], "formatted_output": ""},
            config={"callbacks": [rec]},
        )

    # Run blame analysis
    engine = BlameEngine(rec.session_id)
    result = engine.run()

    if result.blamed_node:
        print(f"Blamed node: {result.blamed_node.node_name}")
        print(f"Reason: {result.reason}")
        print(f"Confidence: {result.confidence}")
    else:
        print("No issues found - all nodes passed!")

    # Show analysis per node
    print("\nPer-node analysis:")
    for analysis in result.analysis:
        status = "X" if analysis.issues_found else "OK"
        print(f"  [{status}] {analysis.node_name}: {analysis.status}")
        for issue in analysis.issues_found:
            print(f"    - {issue}")


if __name__ == "__main__":
    example_recorder()
    example_context_manager()
    example_blame()
    print("\n" + "=" * 60)
    print("Done! Check ~/.langgraph_replay/replays.db for stored sessions.")
    print("=" * 60)
