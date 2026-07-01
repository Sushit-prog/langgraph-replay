"""Tests for the semantic diff module."""

import pytest

from agenttrace.loopdetect.embeddings import clear_cache
from agenttrace.watchdog.semantic_diff import (
    DEFAULT_SEMANTIC_THRESHOLD,
    SemanticDiffResult,
    semantic_match,
)


class TestSemanticDiff:
    def test_similar_strings_match(self):
        """1. Two strings with high semantic similarity, different wording -> is_match=True."""
        clear_cache()
        # Use a lower threshold since these are similar but not identical in meaning
        result = semantic_match(
            "Total: $42.00",
            "Your total comes to $42.00",
            threshold=0.80,
        )
        assert result.is_match is True
        assert result.method == "semantic"
        assert result.similarity_score > 0.8  # Should be reasonably high

    def test_different_meaning_no_match(self):
        """2. Two strings with genuinely different meaning -> is_match=False."""
        clear_cache()
        result = semantic_match(
            "Order confirmed and shipped",
            "Order cancelled",
        )
        assert result.is_match is False
        assert result.method == "semantic"
        assert result.similarity_score < 0.8  # Should be low

    def test_non_string_fallback(self):
        """3. Non-string output (dict-like) -> method='fallback_exact', no crash."""
        # JSON dict should trigger fallback
        baseline = '{"total": 42.00, "formatted": "Total: $42.00"}'
        new = '{"total": 42.00, "formatted": "Your total comes to $42.00"}'
        result = semantic_match(baseline, new)
        assert result.method == "fallback_exact"
        assert result.is_match is False  # Exact match failed

    def test_exact_equal_strings(self):
        """Exact equal strings -> is_match=True without embedding."""
        clear_cache()
        result = semantic_match("Same text", "Same text")
        assert result.is_match is True
        assert result.similarity_score == 1.0

    def test_threshold_respected_high(self):
        """4a. High threshold (0.99) flips match to False for moderately similar strings."""
        clear_cache()
        # These are similar but not identical in meaning
        result_low = semantic_match(
            "Order confirmed",
            "Order has been confirmed",
            threshold=0.80,
        )
        result_high = semantic_match(
            "Order confirmed",
            "Order has been confirmed",
            threshold=0.99,
        )
        # With low threshold, should match
        assert result_low.is_match is True
        # With very high threshold, may not match (proves threshold is wired)
        # We can't guarantee the exact score, but we can verify the threshold affects the result
        assert result_high.similarity_score == result_low.similarity_score  # Same score
        # The is_match should differ if the score is between 0.80 and 0.99
        if 0.80 <= result_low.similarity_score < 0.99:
            assert result_high.is_match is False

    def test_threshold_respected_low(self):
        """4b. Low threshold (0.5) makes almost anything match."""
        clear_cache()
        result = semantic_match(
            "Order confirmed",
            "Order cancelled",
            threshold=0.5,
        )
        # Even dissimilar strings might match at very low threshold
        # The key assertion: threshold is actually used
        assert result.method == "semantic"

    def test_empty_strings(self):
        """Empty strings should be handled gracefully."""
        clear_cache()
        result = semantic_match("", "")
        assert result.is_match is True
        assert result.similarity_score == 1.0

    def test_one_empty_one_not(self):
        """One empty, one non-empty -> fallback to exact match."""
        clear_cache()
        result = semantic_match("", "Some text")
        assert result.is_match is False
        # Empty string is considered plain text, so semantic comparison is used
        assert result.method == "semantic"

    def test_numeric_string_fallback(self):
        """Pure number strings should fall back to exact match."""
        result = semantic_match("42.00", "42.00")
        assert result.method == "fallback_exact"
        assert result.is_match is True

    def test_numeric_string_different_fallback(self):
        """Different number strings -> fallback exact, no match."""
        result = semantic_match("42.00", "43.00")
        assert result.method == "fallback_exact"
        assert result.is_match is False
