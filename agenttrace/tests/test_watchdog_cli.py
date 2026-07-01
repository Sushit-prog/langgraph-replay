"""Tests for the watchdog CLI commands."""

import json
import os
import tempfile

import pytest
from click.testing import CliRunner

from agenttrace.annotations.models import Annotation, Judgment
from agenttrace.annotations.store import AnnotationStore
from agenttrace.watchdog.baseline import BaselineStore
from agenttrace.watchdog.cli import baseline, watchdog
from langgraph_replay.storage import NodeExecution, ReplayStorage, Session

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def _load_fixture(name: str) -> dict:
    with open(os.path.join(FIXTURES_DIR, name)) as f:
        return json.load(f)


def _populate_trace(storage: ReplayStorage, fixture: dict) -> None:
    s = fixture["session"]
    session = Session(
        id=s["id"],
        agent_name=s["agent_name"],
        created_at=s["created_at"],
        total_nodes=s["total_nodes"],
        status=s["status"],
        final_output=s.get("final_output", ""),
        metadata=s.get("metadata", {}),
    )
    storage.save_session(session)
    for e in fixture["executions"]:
        storage.save_node_execution(NodeExecution(
            id=e["id"],
            session_id=e["session_id"],
            node_name=e["node_name"],
            execution_order=e["execution_order"],
            input_state=e.get("input_state", ""),
            output_state=e.get("output_state", ""),
            started_at=e.get("started_at", ""),
            duration_ms=e.get("duration_ms", 0.0),
            status=e.get("status", "success"),
            error_message=e.get("error_message"),
            llm_calls=e.get("llm_calls", 0),
        ))


def _populate_annotations(store: AnnotationStore, fixture: dict) -> None:
    for a in fixture.get("annotations", []):
        store.save(Annotation(
            run_id=a["round_id"] if "round_id" in a else a["run_id"],
            step_id=a["step_id"],
            judgment=Judgment(a["judgment"]),
            note=a.get("note"),
            annotator=a.get("annotator"),
        ))


@pytest.fixture
def env(tmp_path):
    """Set up temp stores and patch env vars for the CLI."""
    ann_db = str(tmp_path / "annotations.db")
    trace_db = str(tmp_path / "replays.db")

    ann_store = AnnotationStore(ann_db)
    trace_store = ReplayStorage(trace_db)

    baseline = _load_fixture("baseline_run_fixture.json")
    _populate_trace(trace_store, baseline)
    _populate_annotations(ann_store, baseline)

    clean = _load_fixture("clean_run_fixture.json")
    _populate_trace(trace_store, clean)

    regressed = _load_fixture("regressed_run_fixture.json")
    _populate_trace(trace_store, regressed)

    ann_store.close()
    trace_store.close()

    # Patch env vars so stores use our temp DBs
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("AGENTTRACE_DB", ann_db)
    monkeypatch.setenv("LANGGRAPH_REPLAY_DB", trace_db)

    yield tmp_path, monkeypatch

    monkeypatch.undo()


class TestBaselineCLI:
    def test_set_and_show_round_trip(self, env):
        """1. baseline set / baseline show round-trip."""
        tmp_path, _ = env
        runner = CliRunner()

        # Set baseline (use a temp dir to avoid polluting project)
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(baseline, ["set", "baseline-run-001"])
            assert result.exit_code == 0
            assert "baseline-run-001" in result.output

            result = runner.invoke(baseline, ["show"])
            assert result.exit_code == 0
            assert "baseline-run-001" in result.output

    def test_show_no_baseline_exits_2(self, env):
        """baseline show with no baseline -> exit code 2."""
        tmp_path, _ = env
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(baseline, ["show"])
            assert result.exit_code == 2
            assert "No baseline pinned" in result.output


class TestWatchCLI:
    def test_watch_no_baseline_exits_2(self, env):
        """2. watch with no baseline pinned and no --baseline -> exit code 2."""
        tmp_path, _ = env
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(watchdog, ["watch", "clean-run-001"])
            assert result.exit_code == 2
            assert "No baseline pinned" in result.output

    def test_watch_clean_run_exits_0(self, env):
        """watch against clean run -> exit 0."""
        tmp_path, _ = env
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(watchdog, [
                "watch", "clean-run-001",
                "--baseline", "baseline-run-001",
                "-o", "report.json",
            ])
            assert result.exit_code == 0
            assert os.path.exists("report.json")

    def test_watch_regressed_run_exits_1(self, env):
        """watch against regressed run -> exit 1."""
        tmp_path, _ = env
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(watchdog, [
                "watch", "regressed-run-001",
                "--baseline", "baseline-run-001",
                "-o", "report.json",
            ])
            assert result.exit_code == 1
            assert os.path.exists("report.json")

    def test_quiet_suppresses_table(self, env):
        """3. --quiet suppresses the human-readable table but writes JSON."""
        tmp_path, _ = env
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(watchdog, [
                "watch", "clean-run-001",
                "--baseline", "baseline-run-001",
                "--quiet",
                "-o", "report.json",
            ])
            assert result.exit_code == 0
            # In quiet mode, the table icons (~, !, ?) should not appear
            assert "~" not in result.output
            assert "!" not in result.output
            assert os.path.exists("report.json")

    def test_json_report_schema_locked(self, env):
        """4. JSON report schema has stable key names."""
        tmp_path, _ = env
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(watchdog, [
                "watch", "clean-run-001",
                "--baseline", "baseline-run-001",
                "-o", "report.json",
            ])

            with open("report.json") as f:
                report = json.load(f)

            expected_top = {"baseline_run_id", "new_run_id", "regression_count", "structural_change_count", "has_regression", "findings"}
            assert set(report.keys()) == expected_top

            expected_finding = {"step_id", "node_name", "judgment", "finding_type", "baseline_output", "new_output", "annotation_note"}
            for finding in report["findings"]:
                assert set(finding.keys()) == expected_finding
