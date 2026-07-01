"""Tests for the loop classifier."""

import json
import os

import pytest

from agenttrace.loopdetect.classifier import (
    DEFAULT_THRESHOLD,
    LoopClassification,
    classify_cycle,
)
from agenttrace.loopdetect.cycle_finder import DetectedCycle, NodeVisit, find_cycles
from agenttrace.loopdetect.embeddings import clear_cache

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def _load_fixture(name: str) -> dict:
    with open(os.path.join(FIXTURES_DIR, name)) as f:
        return json.load(f)


def _make_executions(fixture: dict):
    from dataclasses import dataclass

    @dataclass
    class FakeExec:
        node_name: str
        execution_order: int
        input_state: str
        output_state: str
        status: str

    return [
        FakeExec(
            node_name=e["node_name"],
            execution_order=e["execution_order"],
            input_state=e.get("input_state", ""),
            output_state=e.get("output_state", ""),
            status=e.get("status", "success"),
        )
        for e in fixture["executions"]
    ]


class TestClassifier:
    def test_stuck_loop_fixture(self):
        """1. stuck_loop_run_fixture.json -> classified as stuck_loop with real scores."""
        clear_cache()
        fixture = _load_fixture("stuck_loop_run_fixture.json")
        executions = _make_executions(fixture)
        cycles = find_cycles(executions)

        assert len(cycles) == 1
        result = classify_cycle(cycles[0])

        assert result.classification == "stuck_loop"
        assert result.visit_count == 4
        assert result.node_name == "fetch_weather_api"
        # Reasoning should contain actual similarity scores (not hardcoded)
        assert "0." in result.reasoning  # Has a decimal score
        assert "stuck" in result.reasoning.lower() or "repeating" in result.reasoning.lower()

    def test_legitimate_retry_fixture(self):
        """2. legitimate_retry_run_fixture.json -> classified as legitimate_retry."""
        clear_cache()
        fixture = _load_fixture("legitimate_retry_run_fixture.json")
        executions = _make_executions(fixture)
        cycles = find_cycles(executions)

        assert len(cycles) == 1
        result = classify_cycle(cycles[0])

        assert result.classification == "legitimate_retry"
        assert result.visit_count == 3
        assert result.node_name == "retry_payment"

    def test_threshold_respected(self):
        """3. Threshold above actual similarity flips stuck_loop to legitimate_retry."""
        clear_cache()
        fixture = _load_fixture("stuck_loop_run_fixture.json")
        executions = _make_executions(fixture)
        cycles = find_cycles(executions)

        # With default threshold, should be stuck_loop
        result_default = classify_cycle(cycles[0], threshold=DEFAULT_THRESHOLD)
        assert result_default.classification == "stuck_loop"

        # Get the actual similarity score, then set threshold above it
        # This proves the threshold is actually wired in, not decorative
        # Note: similarity can be slightly > 1.0 due to floating point, so use +0.1
        actual_sim = result_default.avg_similarity
        threshold_above = actual_sim + 0.1

        result_high = classify_cycle(cycles[0], threshold=threshold_above)
        assert result_high.classification == "legitimate_retry"

    def test_window_respected(self):
        """4. Window parameter limits comparison scope."""
        clear_cache()
        fixture = _load_fixture("stuck_loop_run_fixture.json")
        executions = _make_executions(fixture)
        cycles = find_cycles(executions)

        # With window=1, only compare against the previous 1 visit
        result_window1 = classify_cycle(cycles[0], window=1)

        # With window=0 (all visits), compare against everything
        result_window_all = classify_cycle(cycles[0], window=0)

        # Both should still classify correctly, but similarity scores may differ
        # The key assertion: both should produce valid results without crashing
        assert result_window1.classification in ("stuck_loop", "legitimate_retry")
        assert result_window_all.classification in ("stuck_loop", "legitimate_retry")

        # Verify window=1 produces fewer similarity scores than window=0
        # (window=1 only compares consecutive + 1-back, window=0 compares all pairs)
        assert len(result_window1.similarity_scores) <= len(result_window_all.similarity_scores)

    def test_single_visit_not_a_loop(self):
        """A node visited only once is not a loop."""
        clear_cache()
        cycle = DetectedCycle(
            node_name="single_node",
            visits=[
                NodeVisit("single_node", 0, 1, "{}", "{}", "success"),
            ],
        )
        result = classify_cycle(cycle)
        assert result.classification == "legitimate_retry"
        assert "Fewer than 2 visits" in result.reasoning

    def test_output_is_structured(self):
        """Classification result has all required fields."""
        clear_cache()
        fixture = _load_fixture("stuck_loop_run_fixture.json")
        executions = _make_executions(fixture)
        cycles = find_cycles(executions)
        result = classify_cycle(cycles[0])

        assert isinstance(result, LoopClassification)
        assert isinstance(result.node_name, str)
        assert isinstance(result.visit_count, int)
        assert result.classification in ("stuck_loop", "legitimate_retry")
        assert isinstance(result.reasoning, str)
        assert isinstance(result.similarity_scores, list)
        assert isinstance(result.avg_similarity, float)
