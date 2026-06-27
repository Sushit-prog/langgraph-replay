"""Tests for RegressionWatchdog."""

import pytest
from pathlib import Path

from langgraph_replay.storage import Session, ReplayStorage
from langgraph_replay.watchdog import RegressionWatchdog, RegressionReport


class TestRegressionWatchdog:
    """Tests for the watchdog module."""

    def test_file_hash_changes_on_edit(self, tmp_path):
        """Hash changes when file is modified."""
        f = tmp_path / "agent.py"
        f.write_text("x = 1")
        w = RegressionWatchdog(
            agent_file=str(f),
            agent_name="test",
            storage=ReplayStorage(db_path=str(tmp_path / "db.sqlite")),
        )
        h1 = w._file_hash()
        f.write_text("x = 2")
        h2 = w._file_hash()
        assert h1 != h2

    def test_get_baseline_sessions(self, storage):
        """Returns completed session IDs for agent."""
        for i in range(3):
            s = Session(
                id=f"s_{i}",
                agent_name="my_agent",
                created_at=f"2024-01-15T10:00:0{i}Z",
                total_nodes=1,
                status="completed",
            )
            storage.save_session(s)

        w = RegressionWatchdog(
            agent_file="dummy.py",
            agent_name="my_agent",
            storage=storage,
        )
        ids = w._get_baseline_sessions()
        assert len(ids) == 3
        assert all(i.startswith("s_") for i in ids)

    def test_get_baseline_sessions_limit(self, storage):
        """Respects sessions limit."""
        for i in range(10):
            s = Session(
                id=f"sess_{i}",
                agent_name="agent_x",
                created_at=f"2024-01-15T10:00:{i:02d}Z",
                total_nodes=1,
                status="completed",
            )
            storage.save_session(s)

        w = RegressionWatchdog(
            agent_file="dummy.py",
            agent_name="agent_x",
            storage=storage,
            sessions=3,
        )
        ids = w._get_baseline_sessions()
        assert len(ids) == 3

    def test_regression_report_model(self):
        """RegressionReport fields are accessible."""
        r = RegressionReport(
            old_session_id="old_1",
            new_session_id="new_1",
            has_regression=True,
            blamed_node="summarize",
            reason="key dropped",
        )
        assert r.old_session_id == "old_1"
        assert r.has_regression is True
        assert r.blamed_node == "summarize"

    def test_watchdog_stop(self, storage):
        """stop() sets _running to False."""
        w = RegressionWatchdog(
            agent_file="dummy.py",
            agent_name="test",
            storage=storage,
        )
        w._running = True
        w.stop()
        assert w._running is False

    def test_file_hash_missing_file(self, tmp_path):
        """Hash returns empty string for missing file."""
        w = RegressionWatchdog(
            agent_file=str(tmp_path / "nonexistent.py"),
            agent_name="test",
            storage=ReplayStorage(db_path=str(tmp_path / "db.sqlite")),
        )
        assert w._file_hash() == ""

    def test_regression_report_no_regression(self):
        """Report with no regression."""
        r = RegressionReport(
            old_session_id="a",
            new_session_id="b",
            has_regression=False,
        )
        assert r.blamed_node is None
        assert r.reason == ""
