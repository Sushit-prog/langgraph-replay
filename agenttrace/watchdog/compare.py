"""Core comparison logic: diff new run against baseline, using annotations as ground truth."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from agenttrace.annotations.models import Judgment
from agenttrace.annotations.store import AnnotationStore
from agenttrace.watchdog.helpers import build_node_index, find_baseline_exec
from agenttrace.watchdog.semantic_diff import DEFAULT_SEMANTIC_THRESHOLD, semantic_match
from agenttrace.watchdog.upstream import UpstreamDivergence, find_upstream_divergence
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
    semantic_note: Optional[str] = None  # Attached when semantic diff was used
    upstream_divergences: list = field(default_factory=list)  # Phase 5: upstream analysis


@dataclass
class ComparisonResult:
    """Aggregated result of comparing a new run against a baseline."""

    baseline_run_id: str
    new_run_id: str
    findings: list[Finding]
    diff_strategy: str = "exact"  # "exact" or "semantic"
    semantic_threshold: Optional[float] = None  # Only set when diff_strategy="semantic"

    @property
    def regression_count(self) -> int:
        return sum(1 for f in self.findings if f.finding_type == FindingType.REGRESSION)

    @property
    def structural_change_count(self) -> int:
        return sum(1 for f in self.findings if f.finding_type == FindingType.STRUCTURAL_CHANGE)

    @property
    def has_regression(self) -> bool:
        return self.regression_count > 0


def compare_runs(
    baseline_run_id: str,
    new_run_id: str,
    annotation_store: Optional[AnnotationStore] = None,
    trace_store: Optional[ReplayStorage] = None,
    diff_strategy: str = "exact",
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    include_upstream_divergence: bool = False,
) -> ComparisonResult:
    """Compare a new run against a baseline using annotations as ground truth.

    Steps annotated 'correct' or 'expected' in the baseline are checked for
    regression. Steps annotated 'incorrect' or 'unexpected' are ignored —
    those are bug-tracking signals, not regression signals.

    Matching strategy: steps are matched by (node_name, occurrence_index) within
    the run's execution order. If a baseline node doesn't exist in the new run,
    it's reported as a structural_change (not a regression).

    Args:
        diff_strategy: "exact" for string equality (default), "semantic" for
            embedding-based similarity comparison.
        semantic_threshold: Threshold for semantic matching (default 0.90).
            Only used when diff_strategy="semantic".
        include_upstream_divergence: When True, for each regression finding,
            analyze upstream steps to find divergent tool outputs and context.

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
        new_index = build_node_index(new_execs)

        # Track occurrence counts as we iterate baseline (to match the index)
        baseline_counts: dict[str, int] = {}

        findings: list[Finding] = []
        for ann in ground_truth:
            # Determine the occurrence index for this baseline step
            # Use the step's node_name as the match key (step_id is annotation-specific)
            # We need to find which baseline execution corresponds to this step_id
            baseline_exec = find_baseline_exec(ann, baseline_execs, baseline_counts)

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

            if diff_strategy == "semantic":
                # Use semantic comparison
                sem_result = semantic_match(baseline_out, new_out, threshold=semantic_threshold)
                if sem_result.is_match:
                    finding_type = FindingType.UNCHANGED
                    # Attach note explaining that raw strings differed but were
                    # judged semantically equivalent — transparency matters more
                    # than a clean "unchanged" label
                    if baseline_out != new_out:
                        semantic_note = (
                            f"raw output text differs but judged equivalent "
                            f"(similarity={sem_result.similarity_score:.2f})"
                        )
                    else:
                        semantic_note = None
                else:
                    finding_type = FindingType.REGRESSION
                    if sem_result.method == "fallback_exact":
                        semantic_note = "exact-match: non-text field"
                    else:
                        semantic_note = f"similarity={sem_result.similarity_score:.2f}"
            else:
                # Exact comparison (default, unchanged behavior)
                if baseline_out == new_out:
                    finding_type = FindingType.UNCHANGED
                    semantic_note = None
                else:
                    finding_type = FindingType.REGRESSION
                    semantic_note = None

            # Phase 5: upstream divergence analysis for regressions
            upstream_divs = []
            if include_upstream_divergence and finding_type == FindingType.REGRESSION:
                upstream_divs = find_upstream_divergence(
                    baseline_executions=baseline_execs,
                    new_executions=new_execs,
                    regression_step_node_name=node_name,
                    regression_step_execution_order=baseline_exec.execution_order,
                    semantic_threshold=semantic_threshold,
                )

            findings.append(Finding(
                step_id=ann.step_id,
                node_name=node_name,
                judgment=ann.judgment.value,
                finding_type=finding_type,
                baseline_output=baseline_out,
                new_output=new_out,
                annotation_note=ann.note,
                semantic_note=semantic_note,
                upstream_divergences=upstream_divs,
            ))

        return ComparisonResult(
            baseline_run_id=baseline_run_id,
            new_run_id=new_run_id,
            findings=findings,
            diff_strategy=diff_strategy,
            semantic_threshold=semantic_threshold if diff_strategy == "semantic" else None,
        )
    finally:
        if own_stores:
            annotation_store.close()
            trace_store.close()
