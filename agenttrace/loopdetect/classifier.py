"""Classify detected cycles as stuck_loop or legitimate_retry.

Given a DetectedCycle from cycle_finder.py, compute similarity scores between
visits and apply a progress heuristic to classify the loop behavior.
"""

from dataclasses import dataclass, field

from agenttrace.loopdetect.cycle_finder import DetectedCycle, NodeVisit
from agenttrace.loopdetect.embeddings import cosine_similarity, embed

# Default similarity threshold — above this, visits are considered "same state"
DEFAULT_THRESHOLD = 0.92

# Default window — compare each visit against the previous K occurrences
DEFAULT_WINDOW = 3


def _state_text(visit: NodeVisit) -> str:
    """Define what 'state' means for embedding purposes.

    We concatenate input + output text. This captures both what the node
    received and what it produced, giving a full picture of the node's
    behavior at each visit.
    """
    return f"INPUT: {visit.input_state}\nOUTPUT: {visit.output_state}"


@dataclass
class LoopClassification:
    """Result of classifying a detected cycle."""

    node_name: str
    visit_count: int
    classification: str  # "stuck_loop" or "legitimate_retry"
    reasoning: str
    similarity_scores: list[float] = field(default_factory=list)
    avg_similarity: float = 0.0


def classify_cycle(
    cycle: DetectedCycle,
    threshold: float = DEFAULT_THRESHOLD,
    window: int = DEFAULT_WINDOW,
) -> LoopClassification:
    """Classify a detected cycle as stuck_loop or legitimate_retry.

    Args:
        cycle: The detected cycle to classify.
        threshold: Similarity threshold. Visits with avg similarity above this
                   AND no progress signal are classified as stuck_loop.
        window: Compare each visit against the previous K visits, not all.
                Set to 0 to compare against all visits.

    Returns:
        LoopClassification with the verdict and reasoning.
    """
    visits = cycle.visits
    n = len(visits)

    if n < 2:
        return LoopClassification(
            node_name=cycle.node_name,
            visit_count=n,
            classification="legitimate_retry",
            reasoning="Fewer than 2 visits — not a loop.",
        )

    # Compute similarity scores between consecutive visits and within window
    scores: list[float] = []
    for i in range(1, n):
        vec_prev = embed(_state_text(visits[i - 1]))
        vec_curr = embed(_state_text(visits[i]))
        score = cosine_similarity(vec_prev, vec_curr)
        scores.append(score)

    # Also compute similarity against the configured window
    window_scores: list[float] = []
    effective_window = window if window > 0 else n - 1
    for i in range(1, n):
        window_start = max(0, i - effective_window)
        for j in range(window_start, i):
            if i == j + 1:
                continue  # Already computed as consecutive
            vec_j = embed(_state_text(visits[j]))
            vec_i = embed(_state_text(visits[i]))
            window_scores.append(cosine_similarity(vec_j, vec_i))

    all_scores = scores + window_scores
    avg_sim = sum(all_scores) / len(all_scores) if all_scores else 0.0

    # Progress heuristic:
    # If avg similarity is high AND visits don't show meaningful variation
    # (e.g. same input state each time, or same output), it's a stuck loop.
    # We check if any consecutive pair has a score below (threshold - 0.1),
    # which would indicate some state change happened between visits.
    has_variation = any(s < (threshold - 0.1) for s in scores)

    # Also check if the later visits show improvement (output changes meaningfully)
    # by looking at input state similarity (do the inputs change?)
    input_scores: list[float] = []
    for i in range(1, n):
        vec_prev_in = embed(f"IN: {visits[i - 1].input_state}")
        vec_curr_in = embed(f"IN: {visits[i].input_state}")
        input_scores.append(cosine_similarity(vec_prev_in, vec_curr_in))

    avg_input_sim = sum(input_scores) / len(input_scores) if input_scores else 1.0
    inputs_changing = avg_input_sim < threshold

    # Classification logic
    if avg_sim >= threshold and not has_variation and not inputs_changing:
        classification = "stuck_loop"
        reasoning = (
            f"avg similarity {avg_sim:.2f} across {n} visits (threshold: {threshold}), "
            f"no meaningful state variation detected — "
            f"agent appears to be repeating the same action without progress."
        )
    else:
        classification = "legitimate_retry"
        reasons = []
        if avg_sim < threshold:
            reasons.append(f"avg similarity {avg_sim:.2f} is below threshold {threshold}")
        if has_variation:
            reasons.append("state variation detected between visits")
        if inputs_changing:
            reasons.append("input parameters changed between visits")
        reasoning = (
            f"avg similarity {avg_sim:.2f}, "
            + " — ".join(reasons if reasons else ["visits show meaningful progression"])
            + f" — treating as valid retry."
        )

    return LoopClassification(
        node_name=cycle.node_name,
        visit_count=n,
        classification=classification,
        reasoning=reasoning,
        similarity_scores=scores,
        avg_similarity=avg_sim,
    )
