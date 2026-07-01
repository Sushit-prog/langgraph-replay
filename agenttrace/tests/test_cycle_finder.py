"""Tests for the cycle finder."""

import json
import os

import pytest

from agenttrace.loopdetect.cycle_finder import DetectedCycle, find_cycles

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def _load_fixture(name: str) -> dict:
    with open(os.path.join(FIXTURES_DIR, name)) as f:
        return json.load(f)


def _make_executions(fixture: dict):
    """Convert fixture executions into objects with the required attributes."""
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


class TestCycleFinder:
    def test_finds_repeated_node(self):
        """1. Run with a repeated node -> correctly identifies it and orders its visits."""
        fixture = _load_fixture("stuck_loop_run_fixture.json")
        executions = _make_executions(fixture)

        cycles = find_cycles(executions)

        assert len(cycles) == 1
        assert cycles[0].node_name == "fetch_weather_api"
        assert cycles[0].visit_count == 4

        # Visits should be ordered by execution_order
        orders = [v.execution_order for v in cycles[0].visits]
        assert orders == sorted(orders)

    def test_no_repeated_nodes(self):
        """2. Run with no repeated nodes -> returns empty, no false positives."""
        fixture = _load_fixture("clean_run_fixture.json")
        executions = _make_executions(fixture)

        cycles = find_cycles(executions)

        assert len(cycles) == 0

    def test_two_repeated_nodes(self):
        """3. Two different repeated nodes -> both identified independently."""
        from dataclasses import dataclass

        @dataclass
        class FakeExec:
            node_name: str
            execution_order: int
            input_state: str
            output_state: str
            status: str

        executions = [
            FakeExec("node_a", 1, "{}", "{}", "success"),
            FakeExec("node_b", 2, "{}", "{}", "success"),
            FakeExec("node_a", 3, "{}", "{}", "success"),
            FakeExec("node_b", 4, "{}", "{}", "success"),
            FakeExec("node_c", 5, "{}", "{}", "success"),
            FakeExec("node_a", 6, "{}", "{}", "success"),
        ]

        cycles = find_cycles(executions)

        assert len(cycles) == 2
        names = [c.node_name for c in cycles]
        assert "node_a" in names
        assert "node_b" in names
        assert "node_c" not in names

        # node_a should have 3 visits, node_b should have 2
        node_a = next(c for c in cycles if c.node_name == "node_a")
        node_b = next(c for c in cycles if c.node_name == "node_b")
        assert node_a.visit_count == 3
        assert node_b.visit_count == 2

    def test_legitimate_retry_has_repeated_node(self):
        """Legitimate retry run also has repeated nodes (retry_payment)."""
        fixture = _load_fixture("legitimate_retry_run_fixture.json")
        executions = _make_executions(fixture)

        cycles = find_cycles(executions)

        assert len(cycles) == 1
        assert cycles[0].node_name == "retry_payment"
        assert cycles[0].visit_count == 3
