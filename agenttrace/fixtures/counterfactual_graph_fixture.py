"""Small, real, compilable LangGraph graph for counterfactual replay testing.

This graph simulates a refund processing flow:
1. parse_request: Parses the user's refund request
2. lookup_refund_policy: Looks up the refund policy (this is where divergence occurs)
3. decide_refund: Decides whether to approve based on policy
4. summarize_answer: Produces the final answer

The divergence scenario:
- Baseline: policy says "full refund within 30 days" -> refund approved
- New run: policy says "refund requires proof of defect" -> refund denied
- Counterfactual: substituting baseline policy should resolve the regression
"""

from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

# Shared checkpointer for all test runs in this module
_shared_checkpointer = MemorySaver()


class RefundState(TypedDict):
    order_id: int
    request: str
    policy: str
    eligible: bool
    decision: str
    answer: str
    tool_calls_0_output: str  # Simulated tool call output


def parse_request(state: RefundState) -> dict:
    """Parse the refund request."""
    return {"order_id": state.get("order_id", 4471)}


def lookup_refund_policy(state: RefundState) -> dict:
    """Look up refund policy (simulated tool call)."""
    # In real usage, this would call an API
    # The policy text is passed via state to allow counterfactual substitution
    policy = state.get("tool_calls_0_output", "Default policy")
    policy_lower = policy.lower()
    # Check for refund eligibility keywords
    eligible = (
        "full refund" in policy_lower
        or "complete refund" in policy_lower
        or "eligible" in policy_lower
    )
    return {"policy": policy, "eligible": eligible}


def decide_refund(state: RefundState) -> dict:
    """Decide whether to approve refund based on policy."""
    if state.get("eligible"):
        return {"decision": "approved"}
    else:
        return {"decision": "denied"}


def summarize_answer(state: RefundState) -> dict:
    """Produce the final answer based on the decision."""
    decision = state.get("decision", "unknown")
    order_id = state.get("order_id", 0)

    if decision == "approved":
        answer = f"Refund approved for order #{order_id}."
    else:
        answer = f"Refund denied for order #{order_id}."

    return {"answer": answer}


def build_graph():
    """Build and compile the refund processing graph with shared checkpointer."""
    graph = StateGraph(RefundState)

    graph.add_node("parse_request", parse_request)
    graph.add_node("lookup_refund_policy", lookup_refund_policy)
    graph.add_node("decide_refund", decide_refund)
    graph.add_node("summarize_answer", summarize_answer)

    graph.set_entry_point("parse_request")
    graph.add_edge("parse_request", "lookup_refund_policy")
    graph.add_edge("lookup_refund_policy", "decide_refund")
    graph.add_edge("decide_refund", "summarize_answer")
    graph.add_edge("summarize_answer", END)

    # Compile with shared checkpointer for counterfactual replay
    return graph.compile(checkpointer=_shared_checkpointer)


def run_baseline(thread_id: str = None):
    """Run the baseline scenario (refund approved)."""
    graph = build_graph()
    if thread_id is None:
        import uuid
        thread_id = f"baseline-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "order_id": 4471,
        "request": "refund for order 4471",
        "tool_calls_0_output": "All orders within 30 days receive a complete refund with no questions asked.",
    }

    result = graph.invoke(initial_state, config)
    return result, config


def run_new_with_regression(thread_id: str = None):
    """Run the new scenario with regression (refund denied)."""
    graph = build_graph()
    if thread_id is None:
        import uuid
        thread_id = f"new-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "order_id": 4471,
        "request": "refund for order 4471",
        "tool_calls_0_output": "Refunds require documented proof of product defect within 14 days of purchase.",
    }

    result = graph.invoke(initial_state, config)
    return result, config


if __name__ == "__main__":
    # Demo the scenarios
    print("=== BASELINE SCENARIO ===")
    baseline_result, _ = run_baseline()
    print(f"Answer: {baseline_result['answer']}")
    print(f"Policy: {baseline_result['policy']}")
    print()

    print("=== NEW RUN (REGRESSION) ===")
    new_result, _ = run_new_with_regression()
    print(f"Answer: {new_result['answer']}")
    print(f"Policy: {new_result['policy']}")
