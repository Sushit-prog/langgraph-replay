"""Regression watchdog: watches agent files and re-runs sessions on changes."""

import time
import hashlib
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from rich.console import Console

from langgraph_replay.storage import ReplayStorage

console = Console()


class RegressionReport(BaseModel):
    """Result of comparing old vs new session."""
    old_session_id: str
    new_session_id: str
    has_regression: bool
    blamed_node: Optional[str] = None
    reason: str = ""


class RegressionWatchdog:
    """Watches a Python file for changes and re-runs saved sessions.

    Usage::

        watchdog = RegressionWatchdog(
            agent_file="research_agent.py",
            agent_name="research_agent",
            rerun_fn=my_rerun_function,
            sessions=5
        )
        watchdog.start()
    """

    def __init__(
        self,
        agent_file: str,
        agent_name: str,
        rerun_fn=None,
        storage: Optional[ReplayStorage] = None,
        sessions: int = 5,
        poll_interval: float = 2.0,
    ):
        self.agent_file = Path(agent_file)
        self.agent_name = agent_name
        self.rerun_fn = rerun_fn
        self.storage = storage or ReplayStorage()
        self.sessions = sessions
        self.poll_interval = poll_interval
        self._last_hash = self._file_hash()
        self._running = False

    def _file_hash(self) -> str:
        """Returns SHA256 hash of agent file contents."""
        try:
            return hashlib.sha256(
                self.agent_file.read_bytes()
            ).hexdigest()
        except FileNotFoundError:
            return ""

    def _get_baseline_sessions(self) -> list[str]:
        """Returns the last N completed session IDs."""
        sessions = self.storage.search_sessions(
            agent_name=self.agent_name,
            status="completed",
        )
        return [s.id for s in sessions[:self.sessions]]

    def _check_regression(
        self, old_session_id: str, new_session_id: str
    ) -> RegressionReport:
        """Compares old and new sessions."""
        from langgraph_replay.blame import BlameEngine
        engine = BlameEngine(new_session_id, self.storage)
        result = engine.run(
            baseline_session_id=old_session_id,
            use_eval=False,
        )
        return RegressionReport(
            old_session_id=old_session_id,
            new_session_id=new_session_id,
            has_regression=result.blamed_node is not None,
            blamed_node=result.blamed_node.node_name
            if result.blamed_node
            else None,
            reason=result.reason,
        )

    def _on_change(self):
        """Called when file change detected."""
        console.print(
            f"\n[yellow]File change detected: "
            f"{self.agent_file.name}[/yellow]"
        )
        console.print("[blue]Re-running saved sessions...[/blue]")

        baseline_sessions = self._get_baseline_sessions()

        if not baseline_sessions:
            console.print(
                f"[yellow]No saved sessions found for "
                f"agent '{self.agent_name}'[/yellow]"
            )
            return

        regressions = []
        for session_id in baseline_sessions:
            console.print(
                f"[dim]Re-running session {session_id}...[/dim]"
            )
            try:
                if self.rerun_fn:
                    new_id = self.rerun_fn(session_id, self.storage)
                else:
                    new_id = session_id
                report = self._check_regression(session_id, new_id)
                if report.has_regression:
                    regressions.append(report)
                    console.print(
                        f"[red]REGRESSION in {session_id}: "
                        f"{report.blamed_node} -- "
                        f"{report.reason}[/red]"
                    )
                else:
                    console.print(
                        f"[green]OK: {session_id}[/green]"
                    )
            except Exception as e:
                console.print(
                    f"[red]Error re-running {session_id}: "
                    f"{e}[/red]"
                )

        if regressions:
            console.print(
                f"\n[red bold]"
                f"{len(regressions)} regression(s) detected "
                f"after file change.[/red bold]"
            )
        else:
            console.print(
                "\n[green bold]"
                "All sessions passed -- no regressions.[/green bold]"
            )

    def start(self):
        """Start watching. Blocks until Ctrl+C."""
        self._running = True
        console.print(
            f"[green]Watching {self.agent_file.name} "
            f"for changes...[/green]"
        )
        console.print("[dim]Press Ctrl+C to stop[/dim]")

        try:
            while self._running:
                time.sleep(self.poll_interval)
                current_hash = self._file_hash()
                if current_hash != self._last_hash:
                    self._last_hash = current_hash
                    self._on_change()
        except KeyboardInterrupt:
            console.print("\n[yellow]Watchdog stopped.[/yellow]")
            self._running = False

    def stop(self):
        """Stop watching."""
        self._running = False