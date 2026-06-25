"""Diff utilities for comparing states and sessions."""

from typing import Any

from pydantic import BaseModel, Field

from langgraph_replay.storage import NodeExecution, _deserialize_state


class StateDiff(BaseModel):
    """Result of comparing two state dictionaries."""

    added: dict = Field(default_factory=dict)
    removed: dict = Field(default_factory=dict)
    modified: dict = Field(default_factory=dict)
    unchanged: list = Field(default_factory=list)


class NodeComparison(BaseModel):
    """Result of comparing a single node across two sessions."""

    node_name: str
    duration_diff_ms: float
    status_changed: bool
    state_diff: StateDiff


class SessionDiff(BaseModel):
    """Result of comparing two complete sessions."""

    nodes_only_in_a: list = Field(default_factory=list)
    nodes_only_in_b: list = Field(default_factory=list)
    nodes_in_both: list = Field(default_factory=list)


def compute_state_diff(before: dict, after: dict) -> StateDiff:
    """Compare two state dicts and identify changes.

    Args:
        before: The state dict before a node executed.
        after: The state dict after a node executed.

    Returns:
        StateDiff with added, removed, modified, and unchanged keys.
    """
    added = {}
    removed = {}
    modified = {}
    unchanged = []

    all_keys = set(before.keys()) | set(after.keys())

    for key in all_keys:
        if key not in before:
            added[key] = after[key]
        elif key not in after:
            removed[key] = before[key]
        elif before[key] == after[key]:
            unchanged.append(key)
        else:
            modified[key] = {"before": before[key], "after": after[key]}

    return StateDiff(
        added=added,
        removed=removed,
        modified=modified,
        unchanged=unchanged,
    )


def compute_session_diff(
    session_a: list[NodeExecution], session_b: list[NodeExecution]
) -> SessionDiff:
    """Compare two sessions node by node.

    Args:
        session_a: Node executions from the first session.
        session_b: Node executions from the second session.

    Returns:
        SessionDiff with nodes only in each session and per-node comparisons.
    """
    names_a = {e.node_name for e in session_a}
    names_b = {e.node_name for e in session_b}

    nodes_only_in_a = sorted(names_a - names_b)
    nodes_only_in_b = sorted(names_b - names_a)
    common_names = sorted(names_a & names_b)

    nodes_in_both = []
    execs_a = {e.node_name: e for e in session_a}
    execs_b = {e.node_name: e for e in session_b}

    for name in common_names:
        ea = execs_a[name]
        eb = execs_b[name]

        state_a = _deserialize_state(ea.output_state)
        state_b = _deserialize_state(eb.output_state)

        nodes_in_both.append(
            NodeComparison(
                node_name=name,
                duration_diff_ms=round(eb.duration_ms - ea.duration_ms, 2),
                status_changed=ea.status != eb.status,
                state_diff=compute_state_diff(state_a, state_b),
            )
        )

    return SessionDiff(
        nodes_only_in_a=nodes_only_in_a,
        nodes_only_in_b=nodes_only_in_b,
        nodes_in_both=nodes_in_both,
    )
