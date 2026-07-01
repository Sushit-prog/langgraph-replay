"""Upstream divergence detection for regression findings.

When the watchdog flags a regression at a given step, this module diffs the
tool call outputs and retrieved-context sets at that step's upstream (ancestor)
steps between the baseline and new run. This narrows the search space for
causality without attributing it — that's Phase 6's job.

Graph topology: the existing NodeExecution model uses execution_order as the
implicit graph structure. Steps with lower execution_order are ancestors of
steps with higher execution_order. This works for linear chains and simple
DAGs recorded by langgraph-replay.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from agenttrace.watchdog.helpers import build_node_index
from agenttrace.watchdog.semantic_diff import semantic_match


@dataclass
class FieldDiff:
    """Result of diffing a single field between baseline and new run."""

    field_path: str
    changed: bool
    similarity_score: Optional[float]
    baseline_value: Any
    new_value: Any
    note: str


@dataclass
class UpstreamDivergence:
    """A single upstream divergence finding."""

    step_id: str
    node_name: str
    category: str  # "tool_output" or "retrieved_context"
    field_path: str  # e.g. "tool_calls[0].output" or "retrieved_context[2].content"
    changed: bool
    similarity_score: Optional[float]
    baseline_value: Any
    new_value: Any
    note: str  # human-readable explanation

    def to_counterfactual_input(self) -> dict:
        """Serialize to a dict consumable by counterfactual replay CLI.

        Produces a stable schema with all fields Phase 6 needs:
        step_id, node_name, category, field_path, baseline_value, new_value.
        """
        return {
            "step_id": self.step_id,
            "node_name": self.node_name,
            "category": self.category,
            "field_path": self.field_path,
            "baseline_value": self.baseline_value,
            "new_value": self.new_value,
        }

    @classmethod
    def from_counterfactual_input(cls, data: dict) -> "UpstreamDivergence":
        """Deserialize from a dict produced by to_counterfactual_input().

        Fills in changed=True and empty note since those are display-only fields.
        """
        return cls(
            step_id=data["step_id"],
            node_name=data["node_name"],
            category=data["category"],
            field_path=data["field_path"],
            changed=True,
            similarity_score=None,
            baseline_value=data["baseline_value"],
            new_value=data["new_value"],
            note="Loaded from divergence report",
        )


def get_upstream_steps(
    executions: list, target_execution_order: int
) -> list:
    """Return all ancestor steps of a target step by execution_order.

    Uses execution_order as the implicit graph topology: steps with
    execution_order < target are considered ancestors. This covers linear
    chains and simple DAGs recorded by langgraph-replay.

    Returns steps sorted by execution_order (earliest ancestor first).
    """
    return sorted(
        [e for e in executions if e.execution_order < target_execution_order],
        key=lambda e: e.execution_order,
    )


def extract_tool_calls(step) -> list[dict]:
    """Extract tool call information from a step's input/output state.

    Tool calls are identified by looking for 'tool_calls' key in the
    input_state or output_state JSON. Returns a list of dicts with
    'tool_name', 'args', 'output' keys where available.
    """
    tool_calls = []

    # Check input_state for tool call patterns
    try:
        input_data = json.loads(step.input_state) if step.input_state else {}
        if isinstance(input_data, dict) and "tool_calls" in input_data:
            for tc in input_data["tool_calls"]:
                tool_calls.append({
                    "tool_name": tc.get("tool_name", tc.get("name", "unknown")),
                    "args": tc.get("args", tc.get("arguments", {})),
                    "output": None,  # Output not in input
                })
    except (json.JSONDecodeError, TypeError):
        pass

    # Check output_state for tool call results
    try:
        output_data = json.loads(step.output_state) if step.output_state else {}
        if isinstance(output_data, dict) and "tool_calls" in output_data:
            for i, tc in enumerate(output_data["tool_calls"]):
                if i < len(tool_calls):
                    tool_calls[i]["output"] = tc.get("output", tc.get("result"))
                else:
                    tool_calls.append({
                        "tool_name": tc.get("tool_name", tc.get("name", "unknown")),
                        "args": tc.get("args", {}),
                        "output": tc.get("output", tc.get("result")),
                    })
        # Also check for tool_result patterns
        if isinstance(output_data, dict) and "tool_result" in output_data:
            tool_calls.append({
                "tool_name": "tool_result",
                "args": {},
                "output": output_data["tool_result"],
            })
    except (json.JSONDecodeError, TypeError):
        pass

    return tool_calls


def extract_retrieved_context(step) -> list[dict]:
    """Extract retrieved context entries from a step's output state.

    Looks for 'retrieved_context', 'context', 'documents', or 'results'
    keys in the output_state JSON. Returns list of dicts with
    'source' and 'content' keys where available.
    """
    contexts = []

    try:
        output_data = json.loads(step.output_state) if step.output_state else {}
        if not isinstance(output_data, dict):
            return contexts

        # Check multiple possible keys for retrieved context
        for key in ("retrieved_context", "context", "documents", "results", "retrieved_docs"):
            if key in output_data:
                raw = output_data[key]
                if isinstance(raw, list):
                    for item in raw:
                        if isinstance(item, dict):
                            contexts.append({
                                "source": item.get("source", item.get("id", "unknown")),
                                "content": item.get("content", item.get("text", str(item))),
                            })
                        else:
                            contexts.append({
                                "source": "unknown",
                                "content": str(item),
                            })
                elif isinstance(raw, str):
                    contexts.append({
                        "source": "unknown",
                        "content": raw,
                    })
                break  # Only use the first matching key

    except (json.JSONDecodeError, TypeError):
        pass

    return contexts


def diff_field(
    baseline_value: Any,
    new_value: Any,
    semantic_threshold: float,
    field_path: str,
) -> FieldDiff:
    """Compare a single field between baseline and new run.

    If both values are strings: use semantic_match() from Phase 4.
    If either value is non-string: exact equality fallback.
    Missing-in-one-side: reported as changed with appropriate note.
    """
    # Handle missing values
    if baseline_value is None and new_value is None:
        return FieldDiff(
            field_path=field_path,
            changed=False,
            similarity_score=None,
            baseline_value=None,
            new_value=None,
            note="both missing",
        )

    if baseline_value is None:
        return FieldDiff(
            field_path=field_path,
            changed=True,
            similarity_score=None,
            baseline_value=None,
            new_value=new_value,
            note="present in new run only",
        )

    if new_value is None:
        return FieldDiff(
            field_path=field_path,
            changed=True,
            similarity_score=None,
            baseline_value=baseline_value,
            new_value=None,
            note="present in baseline run only",
        )

    # Both values present — compare
    baseline_str = str(baseline_value) if not isinstance(baseline_value, str) else baseline_value
    new_str = str(new_value) if not isinstance(new_value, str) else new_value

    # If both are non-string types, use exact equality
    if not isinstance(baseline_value, str) or not isinstance(new_value, str):
        changed = baseline_value != new_value
        return FieldDiff(
            field_path=field_path,
            changed=changed,
            similarity_score=None,
            baseline_value=baseline_value,
            new_value=new_value,
            note="exact match (non-string values)" if not changed else "values differ",
        )

    # Both are strings — use semantic comparison
    sem_result = semantic_match(baseline_str, new_str, threshold=semantic_threshold)
    return FieldDiff(
        field_path=field_path,
        changed=not sem_result.is_match,
        similarity_score=sem_result.similarity_score,
        baseline_value=baseline_value,
        new_value=new_value,
        note=f"similarity {sem_result.similarity_score:.2f} ({'above' if sem_result.is_match else 'below'} threshold {semantic_threshold})",
    )


def find_upstream_divergence(
    baseline_executions: list,
    new_executions: list,
    regression_step_node_name: str,
    regression_step_execution_order: int,
    semantic_threshold: float = 0.85,
) -> list[UpstreamDivergence]:
    """Find upstream divergences for a regressed step.

    For each ancestor of the regressed step, diffs tool calls and
    retrieved-context entries between baseline and new run.

    Args:
        baseline_executions: All executions from the baseline run.
        new_executions: All executions from the new run.
        regression_step_node_name: Node name of the regressed step.
        regression_step_execution_order: Execution order of the regressed step.
        semantic_threshold: Threshold for semantic comparison of text fields.

    Returns:
        List of UpstreamDivergence records for all compared fields.
    """
    # Get upstream steps in baseline
    baseline_upstream = get_upstream_steps(baseline_executions, regression_step_execution_order)

    if not baseline_upstream:
        return []

    # Build index for new run to match steps
    new_index = build_node_index(new_executions)

    # Track occurrence counts for matching
    baseline_counts: dict[str, int] = {}
    divergences: list[UpstreamDivergence] = []

    for upstream_step in baseline_upstream:
        node_name = upstream_step.node_name
        occurrence = baseline_counts.get(node_name, 0)
        baseline_counts[node_name] = occurrence + 1

        # Find matching step in new run
        new_step = new_index.get((node_name, occurrence + 1))
        if new_step is None:
            # Step doesn't exist in new run — note but don't flag as divergence
            # (structural changes are handled at the finding level, not upstream)
            continue

        # Diff tool calls
        baseline_tools = extract_tool_calls(upstream_step)
        new_tools = extract_tool_calls(new_step)

        # Compare tool calls by position
        max_tools = max(len(baseline_tools), len(new_tools))
        for i in range(max_tools):
            baseline_tc = baseline_tools[i] if i < len(baseline_tools) else None
            new_tc = new_tools[i] if i < len(new_tools) else None

            if baseline_tc is None:
                divergences.append(UpstreamDivergence(
                    step_id=str(upstream_step.execution_order),
                    node_name=node_name,
                    category="tool_output",
                    field_path=f"tool_calls[{i}]",
                    changed=True,
                    similarity_score=None,
                    baseline_value=None,
                    new_value=new_tc,
                    note="tool call present in new run only",
                ))
                continue

            if new_tc is None:
                divergences.append(UpstreamDivergence(
                    step_id=str(upstream_step.execution_order),
                    node_name=node_name,
                    category="tool_output",
                    field_path=f"tool_calls[{i}]",
                    changed=True,
                    similarity_score=None,
                    baseline_value=baseline_tc,
                    new_value=None,
                    note="tool call present in baseline run only",
                ))
                continue

            # Diff tool output (the most important field for causality)
            output_diff = diff_field(
                baseline_tc.get("output"),
                new_tc.get("output"),
                semantic_threshold,
                f"tool_calls[{i}].output",
            )
            if output_diff.changed:
                divergences.append(UpstreamDivergence(
                    step_id=str(upstream_step.execution_order),
                    node_name=node_name,
                    category="tool_output",
                    field_path=output_diff.field_path,
                    changed=True,
                    similarity_score=output_diff.similarity_score,
                    baseline_value=output_diff.baseline_value,
                    new_value=output_diff.new_value,
                    note=output_diff.note,
                ))

        # Diff retrieved context
        baseline_ctx = extract_retrieved_context(upstream_step)
        new_ctx = extract_retrieved_context(new_step)

        max_ctx = max(len(baseline_ctx), len(new_ctx))
        for i in range(max_ctx):
            baseline_entry = baseline_ctx[i] if i < len(baseline_ctx) else None
            new_entry = new_ctx[i] if i < len(new_ctx) else None

            if baseline_entry is None:
                divergences.append(UpstreamDivergence(
                    step_id=str(upstream_step.execution_order),
                    node_name=node_name,
                    category="retrieved_context",
                    field_path=f"retrieved_context[{i}]",
                    changed=True,
                    similarity_score=None,
                    baseline_value=None,
                    new_value=new_entry,
                    note="context entry present in new run only",
                ))
                continue

            if new_entry is None:
                divergences.append(UpstreamDivergence(
                    step_id=str(upstream_step.execution_order),
                    node_name=node_name,
                    category="retrieved_context",
                    field_path=f"retrieved_context[{i}]",
                    changed=True,
                    similarity_score=None,
                    baseline_value=baseline_entry,
                    new_value=None,
                    note="context entry present in baseline run only",
                ))
                continue

            # Diff content field
            content_diff = diff_field(
                baseline_entry.get("content"),
                new_entry.get("content"),
                semantic_threshold,
                f"retrieved_context[{i}].content",
            )
            if content_diff.changed:
                divergences.append(UpstreamDivergence(
                    step_id=str(upstream_step.execution_order),
                    node_name=node_name,
                    category="retrieved_context",
                    field_path=content_diff.field_path,
                    changed=True,
                    similarity_score=content_diff.similarity_score,
                    baseline_value=content_diff.baseline_value,
                    new_value=content_diff.new_value,
                    note=content_diff.note,
                ))

    return divergences
