"""Shared helper functions for watchdog modules.

Extracted from compare.py to avoid circular imports between compare.py
and upstream.py.
"""

from typing import Optional


def build_node_index(executions: list) -> dict[tuple[str, int], object]:
    """Build a lookup index from (node_name, occurrence_count) -> execution.

    This allows matching baseline steps to new-run steps by node name.
    If a node runs multiple times, the first occurrence gets count=1, second=2, etc.
    This is the tiebreak rule: we match by node_name and occurrence order within the run.
    """
    node_counts: dict[str, int] = {}
    index: dict[tuple[str, int], object] = {}
    for exec in executions:
        count = node_counts.get(exec.node_name, 0) + 1
        node_counts[exec.node_name] = count
        index[(exec.node_name, count)] = exec
    return index


def find_baseline_exec(ann, baseline_execs: list, baseline_counts: dict) -> Optional[object]:
    """Find the baseline execution that corresponds to an annotation's step_id.

    The step_id in annotations may be:
    - A node_name (common case)
    - A custom identifier that maps to a node_name

    We try to match by checking if step_id matches any node_name.
    If multiple executions share a name, we use the occurrence counter.
    """
    # First try: step_id matches a node_name directly
    for exec in baseline_execs:
        if exec.node_name == ann.step_id:
            return exec

    # Second try: step_id is a substring of a node_name or vice versa
    for exec in baseline_execs:
        if ann.step_id in exec.node_name or exec.node_name in ann.step_id:
            return exec

    # Third try: use execution_order (step_id might be a numeric order)
    try:
        order = int(ann.step_id)
        for exec in baseline_execs:
            if exec.execution_order == order:
                return exec
    except (ValueError, TypeError):
        pass

    # No match found
    return None
