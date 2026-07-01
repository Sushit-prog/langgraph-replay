"""Core comparison logic: diff new run against baseline, using annotations as ground truth."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from agenttrace.annotations.models import Judgment
from agenttrace.annotations.store import AnnotationStore
from langgraph_replay.storage import ReplayStorage


class FindingType(str, Enum):
    """Classification of a comparison finding."""

    REGRESSION = "regression"
    STRUCTURAL_CHANGE = "structural_change"
    UNCHANGED = "unchanged"


@dataclass
class Finding:
    """A single comparison finding for one step."""

    step_id: str
    node_name: str
    judgment: str
    finding_type: FindingType
    baseline_output: Optional[str] = None
    new_output: Optional[str] = None
    annotation_note: Optional[str] = None


@dataclass
class ComparisonResult:
    """Aggregated result of comparing a new run against a baseline."""

    baseline_run_id: str
    new_run_id: str
    findings: list[Finding]

    @property
    def regression_count(self) -> int:
        return sum(1 for f in self.findings if f.finding_type == FindingType.REGRESSION)

    @property
    def structural_change_count(self) -> int:
        return sum(1 for f in self.findings if f.finding_type == FindingType.STRUCTURAL_CHANGE)

    @property
    def has_regression(self) -> bool:
        return self.regression_count > 0


def _build_node_index(executions: list) -> dict[tuple[str, int], object]:
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


def compare_runs(
    baseline_run_id: str,
    new_run_id: str,
    annotation_store: Optional[AnnotationStore] = None,
    trace_store: Optional[ReplayStorage] = None,
) -> ComparisonResult:
    """Compare a new run against a baseline using annotations as ground truth.

    Steps annotated 'correct' or 'expected' in the baseline are checked for
    regression. Steps annotated 'incorrect' or 'unexpected' are ignored —
    those are bug-tracking signals, not regression signals.

    Matching strategy: steps are matched by (node_name, occurrence_index) within
    the run's execution order. If a baseline node doesn't exist in the new run,
    it's reported as a structural_change (not a regression).

    Returns ComparisonResult with findings for each annotated step.
    """
    own_stores = False
    if annotation_store is None:
        annotation_store = AnnotationStore()
        own_stores = True
    if trace_store is None:
        trace_store = ReplayStorage()
        own_stores = True

    try:
        # Load baseline annotations, filter to correct/expected only
        all_annotations = annotation_store.list_by_run(baseline_run_id)
        ground_truth = [
            a for a in all_annotations
            if a.judgment in (Judgment.CORRECT, Judgment.EXPECTED)
        ]

        if not ground_truth:
            raise ValueError(
                f"No ground-truth annotations found for baseline run {baseline_run_id!r}. "
                "Annotate at least one step as 'correct' or 'expected' before using watch."
            )

        # Load trace executions for both runs
        baseline_execs = trace_store.get_node_executions(baseline_run_id)
        new_execs = trace_store.get_node_executions(new_run_id)

        # Build index for the new run: (node_name, occurrence_index) -> execution
        new_index = _build_node_index(new_execs)

        # Track occurrence counts as we iterate baseline (to match the index)
        baseline_counts: dict[str, int] = {}

        findings: list[Finding] = []
        for ann in ground_truth:
            # Determine the occurrence index for this baseline step
            # Use the step's node_name as the match key (step_id is annotation-specific)
            # We need to find which baseline execution corresponds to this step_id
            baseline_exec = _find_baseline_exec(ann, baseline_execs, baseline_counts)

            if baseline_exec is None:
                # Can't find the corresponding execution — treat as structural change
                findings.append(Finding(
                    step_id=ann.step_id,
                    node_name=ann.step_id,
                    judgment=ann.judgment.value,
                    finding_type=FindingType.STRUCTURAL_CHANGE,
                    annotation_note=ann.note,
                ))
                continue

            node_name = baseline_exec.node_name
            occurrence = baseline_counts.get(node_name, 0)
            baseline_counts[node_name] = occurrence + 1

            # Find matching execution in new run
            new_exec = new_index.get((node_name, occurrence + 1))

            if new_exec is None:
                # Node removed or graph took different branch
                findings.append(Finding(
                    step_id=ann.step_id,
                    node_name=node_name,
                    judgment=ann.judgment.value,
                    finding_type=FindingType.STRUCTURAL_CHANGE,
                    baseline_output=baseline_exec.output_state or "",
                    annotation_note=ann.note,
                ))
                continue

            # Compare outputs
            baseline_out = baseline_exec.output_state or ""
            new_out = new_exec.output_state or ""

            if baseline_out == new_out:
                finding_type = FindingType.UNCHANGED
            else:
                finding_type = FindingType.REGRESSION

            findings.append(Finding(
                step_id=ann.step_id,
                node_name=node_name,
                judgment=ann.judgment.value,
                finding_type=finding_type,
                baseline_output=baseline_out,
                new_output=new_out,
                annotation_note=ann.note,
            ))

        return ComparisonResult(
            baseline_run_id=baseline_run_id,
            new_run_id=new_run_id,
            findings=findings,
        )
    finally:
        if own_stores:
            annotation_store.close()
            trace_store.close()


def _find_baseline_exec(ann, baseline_execs: list, baseline_counts: dict) -> Optional[object]:
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
