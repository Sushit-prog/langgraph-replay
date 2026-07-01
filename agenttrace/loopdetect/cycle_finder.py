"""Detect repeated node visits in a run's execution graph.

Pure graph/sequence analysis — no embeddings here. Separated from classifier.py
so it's independently testable.
"""

from dataclasses import dataclass


@dataclass
class NodeVisit:
    """A single visit to a node, capturing its execution state."""

    node_name: str
    visit_index: int  # 0-based occurrence count for this node
    execution_order: int  # Position in the overall run sequence
    input_state: str  # Raw JSON string of node input
    output_state: str  # Raw JSON string of node output
    status: str  # success/error/etc


@dataclass
class DetectedCycle:
    """A detected cycle: a node visited multiple times."""

    node_name: str
    visits: list[NodeVisit]

    @property
    def visit_count(self) -> int:
        return len(self.visits)


def find_cycles(executions: list) -> list[DetectedCycle]:
    """Walk a run's step sequence and find nodes visited more than once.

    Args:
        executions: List of NodeExecution objects (or similar) with node_name,
                    execution_order, input_state, output_state, status attributes.

    Returns:
        List of DetectedCycle objects, one per repeated node, ordered by
        first occurrence in the run.
    """
    # Group visits by node name, preserving execution order
    node_visits: dict[str, list[NodeVisit]] = {}
    node_first_occurrence: dict[str, int] = {}

    for exec in executions:
        name = exec.node_name
        visit_index = len(node_visits.get(name, []))

        visit = NodeVisit(
            node_name=name,
            visit_index=visit_index,
            execution_order=exec.execution_order,
            input_state=exec.input_state or "",
            output_state=exec.output_state or "",
            status=exec.status,
        )

        if name not in node_visits:
            node_visits[name] = []
            node_first_occurrence[name] = exec.execution_order
        node_visits[name].append(visit)

    # Return only nodes with more than one visit, ordered by first occurrence
    cycles = []
    for name, visits in node_visits.items():
        if len(visits) > 1:
            cycles.append(DetectedCycle(node_name=name, visits=visits))

    cycles.sort(key=lambda c: node_first_occurrence[c.node_name])
    return cycles
