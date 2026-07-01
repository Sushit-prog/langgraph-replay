"""Semantic comparison logic for watchdog outputs.

Thin layer over Phase 3's embeddings (embed(), cosine_similarity()) that
determines whether two output strings are semantically equivalent even if
they differ textually.

Fallback behavior: non-string outputs (dicts, lists, numbers) fall back to
exact-match comparison. This is a documented limitation — embedding a
stringified dict produces meaningless similarity scores, so we don't attempt it.
"""

from dataclasses import dataclass

from agenttrace.loopdetect.embeddings import cosine_similarity, embed

# Default threshold for semantic matching.
# 0.90 is slightly more permissive than the loop classifier's default (0.92)
# because we're measuring "same meaning" not "same state" — wording changes
# that preserve meaning are acceptable here, whereas in loop detection we
# want tighter matching.
DEFAULT_SEMANTIC_THRESHOLD = 0.90


@dataclass
class SemanticDiffResult:
    """Result of a semantic comparison between two outputs."""

    similarity_score: float
    is_match: bool  # True if similarity >= threshold
    method: str  # "semantic" or "fallback_exact"


def _is_plain_text(value: str) -> bool:
    """Determine if a string value is 'plain text' suitable for embedding.

    Returns False for values that look like structured data (JSON dicts/lists,
    pure numbers, booleans) — these should fall back to exact match because
    embedding them produces meaningless similarity scores.

    This is a heuristic boundary, not a hard rule. We err on the side of
    fallback for structured-looking content.
    """
    stripped = value.strip()
    if not stripped:
        return True  # Empty strings are fine to embed

    # JSON-like structures: starts with { or [
    if stripped[0] in ("{", "["):
        return False

    # Pure numbers: try to parse as int/float
    try:
        float(stripped)
        return False
    except ValueError:
        pass

    # Booleans
    if stripped.lower() in ("true", "false", "yes", "no"):
        return False

    return True


def semantic_match(
    baseline_output: str,
    new_output: str,
    threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
) -> SemanticDiffResult:
    """Compare two outputs semantically, with fallback for non-text content.

    Args:
        baseline_output: The baseline step's output (raw JSON string from trace).
        new_output: The new run's step output.
        threshold: Similarity threshold (default 0.90).

    Returns:
        SemanticDiffResult with similarity score, match verdict, and method used.
    """
    # If outputs are exactly equal, no need for semantic comparison
    if baseline_output == new_output:
        # Determine method based on content type
        method = "semantic" if _is_plain_text(baseline_output) else "fallback_exact"
        return SemanticDiffResult(
            similarity_score=1.0,
            is_match=True,
            method=method,
        )

    # Check if outputs are suitable for semantic comparison
    # Fall back to exact match for structured/non-text content
    if not _is_plain_text(baseline_output) or not _is_plain_text(new_output):
        return SemanticDiffResult(
            similarity_score=0.0,
            is_match=False,
            method="fallback_exact",
        )

    # Perform semantic comparison
    vec_a = embed(baseline_output)
    vec_b = embed(new_output)
    score = cosine_similarity(vec_a, vec_b)

    return SemanticDiffResult(
        similarity_score=score,
        is_match=score >= threshold,
        method="semantic",
    )
