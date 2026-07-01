"""Counterfactual replay logic for testing causal hypotheses.

Given a regression flagged by the watchdog and a candidate upstream divergence,
this module replays the new run from a forked checkpoint with the baseline's
tool output substituted in. If the regression disappears, the upstream
divergence is causally implicated.

Known limitation: nodes are assumed idempotent per LangGraph's guidance.
Non-idempotent side effects (e.g. payments, emails) are not guarded against.
"""

import importlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from agenttrace.watchdog.semantic_diff import semantic_match
from agenttrace.watchdog.upstream import UpstreamDivergence


@dataclass
class CounterfactualResult:
    """Result of a counterfactual replay test."""

    divergence_tested: UpstreamDivergence
    regression_resolved: bool
    replayed_output: str
    original_new_output: str
    baseline_output: str
    similarity_to_baseline: float
    note: str


def load_graph_for_run(graph_path: str):
    """Load and compile a LangGraph graph from a module path.

    Args:
        graph_path: Dotted path to a function that returns a compiled graph,
                    e.g. "my_agent.graph:build_graph" or "my_agent:graph".

    Returns:
        A compiled LangGraph graph with a checkpointer.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the function doesn't exist in the module.
        ValueError: If the graph doesn't have a checkpointer.
    """
    # Parse module:function or module.attribute
    if ":" in graph_path:
        module_path, attr_name = graph_path.rsplit(":", 1)
    else:
        # Try to split on last dot
        parts = graph_path.rsplit(".", 1)
        if len(parts) == 2:
            module_path, attr_name = parts
        else:
            raise ValueError(
                f"Invalid graph path: {graph_path!r}. "
                "Expected format: 'module.path:function_name' or 'module.path:graph_attribute'."
            )

    # Import the module
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Could not import module {module_path!r}: {e}. "
            "Ensure the module is installed and accessible."
        ) from e

    # Get the attribute
    try:
        attr = getattr(module, attr_name)
    except AttributeError:
        raise AttributeError(
            f"Module {module_path!r} has no attribute {attr_name!r}. "
            f"Available: {[a for a in dir(module) if not a.startswith('_')]}"
        )

    # Call if it's a function, otherwise use directly
    if callable(attr):
        graph = attr()
    else:
        graph = attr

    # Verify it has a checkpointer
    if not hasattr(graph, "checkpointer") or graph.checkpointer is None:
        raise ValueError(
            "Graph does not have a checkpointer. "
            "Compile with checkpointer=MemorySaver() to enable counterfactual replay."
        )

    return graph


def build_counterfactual_config(
    graph,
    new_run_thread_id: str,
    step_execution_order: int,
) -> dict:
    """Build LangGraph config for forking from a specific checkpoint.

    Args:
        graph: The compiled LangGraph graph.
        new_run_thread_id: The thread_id of the new run to fork from.
        step_execution_order: The execution order of the step to fork at
                              (1-indexed, matching NodeExecution.execution_order).

    Returns:
        Config dict with the correct checkpoint_id for the fork point.
    """
    config = {"configurable": {"thread_id": new_run_thread_id}}

    # Get state history to find the checkpoint at the right step
    # History is newest-first; each entry has metadata["step"] (0-indexed)
    history = list(graph.get_state_history(config))

    if not history:
        raise ValueError(
            f"No checkpoints found for thread_id={new_run_thread_id!r}. "
            "Ensure the run was executed with a checkpointer."
        )

    # Convert execution_order (1-indexed) to step index (0-indexed)
    # execution_order=1 means after the first node, which is step=1 in LangGraph
    target_step = step_execution_order

    # Find the checkpoint with the matching step
    target_checkpoint = None
    for snapshot in history:
        if snapshot.metadata and snapshot.metadata.get("step") == target_step:
            target_checkpoint = snapshot
            break

    if target_checkpoint is None:
        available_steps = [
            s.metadata.get("step") for s in history if s.metadata
        ]
        raise ValueError(
            f"Could not find checkpoint for step={target_step}. "
            f"Available steps: {sorted(set(available_steps))}"
        )

    # Return the config with the checkpoint_id
    return target_checkpoint.config


def inject_baseline_value(
    graph,
    fork_config: dict,
    divergence: UpstreamDivergence,
) -> dict:
    """Fork state at a checkpoint, substituting baseline value for the divergent field.

    Args:
        graph: The compiled LangGraph graph.
        fork_config: Config pointing to the checkpoint to fork from.
        divergence: The upstream divergence to test.

    Returns:
        New config pointing to the forked checkpoint.

    Known limitation: This function needs to know how the upstream node processes
    its inputs to compute derived state values. For production use, this would
    require either:
    1. The user to provide all affected state values, or
    2. A node-specific adapter that knows how to compute outputs from inputs.

    For this implementation, we handle common patterns for tool outputs and
    retrieved context, and fall back to direct injection for other cases.
    """
    baseline_value = divergence.baseline_value

    # Map field_path to state updates based on category
    if divergence.category == "tool_output":
        import re
        match = re.match(r"tool_calls\[(\d+)\]\.(\w+)", divergence.field_path)
        if match:
            idx = int(match.group(1))
            field = match.group(2)
            input_key = f"tool_calls_{idx}_{field}"
        else:
            input_key = divergence.node_name

        # For tool outputs, we inject the input value AND compute derived values
        # This simulates what the node function would produce
        values = {input_key: baseline_value}

        # Compute common derived values based on the input
        # This is a heuristic for common patterns; real-world use would need
        # node-specific adapters or explicit user-provided values
        baseline_str = str(baseline_value).lower()
        if "refund" in baseline_str or "eligible" in baseline_str:
            # Pattern: refund policy lookup
            eligible = (
                "full refund" in baseline_str
                or "complete refund" in baseline_str
                or "eligible" in baseline_str
            )
            values["policy"] = baseline_value
            values["eligible"] = eligible
        elif "price" in baseline_str or "cost" in baseline_str:
            # Pattern: price/cost lookup
            values["price_info"] = baseline_value

    elif divergence.category == "retrieved_context":
        import re
        match = re.match(r"retrieved_context\[(\d+)\]\.(\w+)", divergence.field_path)
        if match:
            idx = int(match.group(1))
            field = match.group(2)
            state_key = f"context_{idx}_{field}"
        else:
            state_key = "context"
        values = {state_key: baseline_value}
    else:
        state_key = divergence.field_path.split("[")[0].split(".")[0]
        values = {state_key: baseline_value}

    # Update state at the fork point
    new_config = graph.update_state(fork_config, values)

    return new_config


def replay_forward(graph, forked_config: dict) -> dict:
    """Resume execution from a forked checkpoint to completion.

    Args:
        graph: The compiled LangGraph graph.
        forked_config: Config pointing to the forked checkpoint.

    Returns:
        The final state after replay.
    """
    # Invoke the graph from the forked checkpoint
    # Use None as input since we're resuming from checkpoint
    result = graph.invoke(None, forked_config)
    return result


def check_regression_resolved(
    original_regression_finding,
    replay_result: dict,
    baseline_output: str,
    semantic_threshold: float = 0.85,
) -> tuple[bool, float, str]:
    """Check if the regression was resolved by the counterfactual substitution.

    Args:
        original_regression_finding: The original Finding from the watchdog.
        replay_result: The state after counterfactual replay.
        baseline_output: The baseline's output at the regression step.
        semantic_threshold: Threshold for semantic matching.

    Returns:
        Tuple of (resolved, similarity_score, note).
    """
    # Extract the output at the regression step from the replay result
    # The output should be in the state under the node's output key
    node_name = original_regression_finding.node_name

    # Try to find the output in the replay result
    replayed_output = ""
    if isinstance(replay_result, dict):
        # Check common output keys - try node name and common suffixes
        # e.g., "summarize_answer" -> try "answer", "summarize_answer", "output", etc.
        possible_keys = [
            node_name,
            node_name.split("_")[-1],  # "answer" from "summarize_answer"
            "answer",
            "output",
            "result",
            "final_output",
            "summary",
        ]
        for key in possible_keys:
            if key in replay_result:
                val = replay_result[key]
                replayed_output = str(val) if val is not None else ""
                break

        # If not found, check if the node name appears in any key
        if not replayed_output:
            for key, val in replay_result.items():
                if isinstance(val, str) and (
                    node_name.lower() in key.lower()
                    or key.lower() in node_name.lower()
                ):
                    replayed_output = val
                    break

    if not replayed_output:
        return False, 0.0, f"Could not extract output for node {node_name!r} from replay result."

    # Compare with baseline using semantic matching
    sem_result = semantic_match(
        replayed_output,
        baseline_output,
        threshold=semantic_threshold,
    )

    if sem_result.is_match:
        note = (
            f"Replayed output matches baseline (similarity={sem_result.similarity_score:.2f} >= {semantic_threshold}). "
            f"Regression resolved."
        )
    else:
        note = (
            f"Replayed output differs from baseline (similarity={sem_result.similarity_score:.2f} < {semantic_threshold}). "
            f"Regression NOT resolved."
        )

    return sem_result.is_match, sem_result.similarity_score, note


def run_counterfactual_test(
    graph_path: str,
    new_run_thread_id: str,
    divergence: UpstreamDivergence,
    regression_finding,
    baseline_output: str,
    semantic_threshold: float = 0.85,
) -> CounterfactualResult:
    """Run a full counterfactual replay test.

    This is the main entry point that orchestrates the entire flow.

    Args:
        graph_path: Module path to the graph builder function.
        new_run_thread_id: Thread ID of the new run to fork from.
        divergence: The upstream divergence to test.
        regression_finding: The original regression finding.
        baseline_output: The baseline's output at the regression step.
        semantic_threshold: Threshold for semantic matching.

    Returns:
        CounterfactualResult with the test outcome.
    """
    # Load the graph
    graph = load_graph_for_run(graph_path)

    # Build config for forking
    fork_config = build_counterfactual_config(
        graph,
        new_run_thread_id,
        step_execution_order=int(divergence.step_id),
    )

    # Inject baseline value
    new_config = inject_baseline_value(graph, fork_config, divergence)

    # Replay forward
    replay_result = replay_forward(graph, new_config)

    # Check if regression resolved
    resolved, similarity, note = check_regression_resolved(
        regression_finding,
        replay_result,
        baseline_output,
        semantic_threshold,
    )

    # Extract replayed output for reporting
    replayed_output = ""
    node_name = regression_finding.node_name
    if isinstance(replay_result, dict):
        for key in (node_name, "output", "result", "final_output", "summary"):
            if key in replay_result:
                replayed_output = str(replay_result[key])
                break

    return CounterfactualResult(
        divergence_tested=divergence,
        regression_resolved=resolved,
        replayed_output=replayed_output,
        original_new_output=regression_finding.new_output or "",
        baseline_output=baseline_output,
        similarity_to_baseline=similarity,
        note=note,
    )
