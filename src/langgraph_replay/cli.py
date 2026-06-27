"""Click-based CLI for langgraph-replay."""

import json
import sys
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from langgraph_replay.blame import BlameEngine
from langgraph_replay.diff import compute_session_diff, _deserialize_state
from langgraph_replay.storage import ReplayStorage

console = Console()


def _get_storage() -> ReplayStorage:
    """Get a ReplayStorage instance."""
    return ReplayStorage()


@click.group()
@click.version_option(version="0.1.0", prog_name="langgraph-replay")
def main() -> None:
    """LangGraph Replay: Record, replay, and debug LangGraph agent executions."""
    pass


@main.command()
@click.option("--limit", default=20, help="Maximum number of sessions to show.")
@click.option("--status", type=click.Choice(["completed", "failed", "interrupted"]), help="Filter by status.")
def list(limit: int, status: Optional[str]) -> None:
    """List recent recorded sessions."""
    try:
        storage = _get_storage()
        if status:
            sessions = storage.search_sessions(status=status)
            sessions = sessions[:limit]
        else:
            sessions = storage.list_sessions(limit=limit)
        storage.close()

        if not sessions:
            console.print("[yellow]No sessions found.[/yellow]")
            return

        table = Table(title="Recorded Sessions")
        table.add_column("Session ID", style="cyan")
        table.add_column("Agent Name")
        table.add_column("Created At")
        table.add_column("Nodes", justify="right")
        table.add_column("Status")

        for session in sessions:
            status_style = "green" if session.status == "completed" else "red"
            table.add_row(
                session.id,
                session.agent_name,
                session.created_at[:19],
                str(session.total_nodes),
                f"[{status_style}]{session.status}[/{status_style}]",
            )

        console.print(table)
    except Exception as e:
        console.print(Panel(f"[red]Error: {e}[/red]", title="Error", border_style="red"))
        sys.exit(1)


@main.command()
@click.argument("session_id")
def show(session_id: str) -> None:
    """Show details for a specific session."""
    try:
        storage = _get_storage()
        session = storage.get_session(session_id)
        if session is None:
            console.print(f"[red]Session '{session_id}' not found.[/red]")
            storage.close()
            sys.exit(1)

        executions = storage.get_node_executions(session_id)
        storage.close()

        # Session info panel
        status_style = "green" if session.status == "completed" else "red"
        info = (
            f"[bold]Agent:[/bold] {session.agent_name}\n"
            f"[bold]Created:[/bold] {session.created_at[:19]}\n"
            f"[bold]Nodes:[/bold] {session.total_nodes}\n"
            f"[bold]Status:[/bold] [{status_style}]{session.status}[/{status_style}]"
        )
        console.print(Panel(info, title=f"Session: {session.id}", border_style="blue"))

        # Node executions table
        table = Table(title="Node Executions")
        table.add_column("Order", justify="right")
        table.add_column("Node Name")
        table.add_column("Duration (ms)", justify="right")
        table.add_column("Status")
        table.add_column("State Keys Changed")

        for exec in executions:
            status_color = "green" if exec.status == "success" else "red"
            input_state = _deserialize_state(exec.input_state)
            output_state = _deserialize_state(exec.output_state)
            changed = set(output_state.keys()) - set(input_state.keys())
            changed_str = ", ".join(changed) if changed else "-"

            table.add_row(
                str(exec.execution_order),
                exec.node_name,
                f"{exec.duration_ms:.1f}",
                f"[{status_color}]{exec.status}[/{status_color}]",
                changed_str,
            )

        console.print(table)
    except SystemExit:
        raise
    except Exception as e:
        console.print(Panel(f"[red]Error: {e}[/red]", title="Error", border_style="red"))
        sys.exit(1)


@main.command()
@click.argument("session_id")
def debug(session_id: str) -> None:
    """Launch the Textual TUI debugger."""
    try:
        storage = _get_storage()
        session = storage.get_session(session_id)
        if session is None:
            console.print(f"[red]Session '{session_id}' not found.[/red]")
            storage.close()
            sys.exit(1)
        storage.close()

        from langgraph_replay.tui.app import DebugApp

        app = DebugApp(session_id)
        app.run()
    except SystemExit:
        raise
    except Exception as e:
        console.print(Panel(f"[red]Error: {e}[/red]", title="Error", border_style="red"))
        sys.exit(1)


@main.command()
@click.argument("session_id_a")
@click.argument("session_id_b")
def diff(session_id_a: str, session_id_b: str) -> None:
    """Compare two sessions."""
    try:
        storage = _get_storage()
        session_a = storage.get_session(session_id_a)
        session_b = storage.get_session(session_id_b)

        if session_a is None:
            console.print(f"[red]Session '{session_id_a}' not found.[/red]")
            storage.close()
            sys.exit(1)
        if session_b is None:
            console.print(f"[red]Session '{session_id_b}' not found.[/red]")
            storage.close()
            sys.exit(1)

        execs_a = storage.get_node_executions(session_id_a)
        execs_b = storage.get_node_executions(session_id_b)
        storage.close()

        session_diff = compute_session_diff(execs_a, execs_b)

        console.print(Panel(
            f"[bold]Session A:[/bold] {session_id_a} ({session_a.agent_name})\n"
            f"[bold]Session B:[/bold] {session_id_b} ({session_b.agent_name})",
            title="Session Diff",
            border_style="blue",
        ))

        if session_diff.nodes_only_in_a:
            console.print(f"\n[red]Nodes only in A:[/red] {', '.join(session_diff.nodes_only_in_a)}")

        if session_diff.nodes_only_in_b:
            console.print(f"\n[green]Nodes only in B:[/green] {', '.join(session_diff.nodes_only_in_b)}")

        if session_diff.nodes_in_both:
            table = Table(title="Node Comparison")
            table.add_column("Node")
            table.add_column("Duration Diff (ms)", justify="right")
            table.add_column("Status Changed")
            table.add_column("State Changes")

            for comp in session_diff.nodes_in_both:
                duration_style = "green" if comp.duration_diff_ms < 0 else "red" if comp.duration_diff_ms > 0 else ""
                changed = len(comp.state_diff.added) + len(comp.state_diff.removed) + len(comp.state_diff.modified)
                table.add_row(
                    comp.node_name,
                    f"[{duration_style}]{comp.duration_diff_ms:+.1f}[/{duration_style}]" if duration_style else f"{comp.duration_diff_ms:+.1f}",
                    "[red]Yes[/red]" if comp.status_changed else "No",
                    str(changed),
                )

            console.print(table)
    except SystemExit:
        raise
    except Exception as e:
        console.print(Panel(f"[red]Error: {e}[/red]", title="Error", border_style="red"))
        sys.exit(1)


@main.command()
@click.argument("session_id")
@click.option("--baseline", help="Baseline session ID. If not provided with --eval, uses most recent completed session for the same agent.")
@click.option("--eval", "use_eval", is_flag=True, help="Use pytest-llm semantic assertions.")
@click.option("--diagnose", is_flag=True, help="Generate root cause and fix suggestions via LLM.")
def blame(session_id: str, baseline: Optional[str], use_eval: bool, diagnose: bool) -> None:
    """Run blame analysis on a session."""
    try:
        storage = _get_storage()
        session = storage.get_session(session_id)
        if session is None:
            console.print(f"[red]Session '{session_id}' not found.[/red]")
            storage.close()
            sys.exit(1)
        storage.close()

        engine = BlameEngine(session_id)
        result = engine.run(baseline_session_id=baseline, use_eval=use_eval, diagnose=diagnose)

        # Build blame output
        lines = []
        for analysis in result.analysis:
            icon = "[green]OK[/green]" if not analysis.issues_found else "[red]X[/red]"
            blame_marker = " [bold red]<-- BLAMED[/bold red]" if analysis.is_blamed else ""
            lines.append(f"{icon} {analysis.node_name}{blame_marker}")
            for issue in analysis.issues_found:
                lines.append(f"    [dim]{issue}[/dim]")

        content = "\n".join(lines)

        if result.blamed_node:
            header = (
                f"[bold red]Blamed Node: {result.blamed_node.node_name}[/bold red]\n"
                f"[red]Reason: {result.reason}[/red]\n"
                f"Confidence: {result.confidence.upper()}\n\n"
            )
            content = header + content
        else:
            content = "[bold green]No issues found[/bold green]\n\n" + content

        console.print(Panel(content, title="Blame Analysis", border_style="red" if result.blamed_node else "green"))

        # Print diagnosis if available
        if result.diagnosis:
            diag = result.diagnosis
            if diag.root_cause and diag.root_cause != "No issues detected":
                console.print(Panel(
                    diag.root_cause,
                    title="Why it broke",
                    border_style="yellow",
                ))
            if diag.fix_suggestions:
                fixes = "\n".join(
                    f"[{i + 1}] {s}" for i, s in enumerate(diag.fix_suggestions)
                )
                console.print(Panel(
                    fixes,
                    title="How to fix it",
                    border_style="green",
                ))
    except SystemExit:
        raise
    except Exception as e:
        console.print(Panel(f"[red]Error: {e}[/red]", title="Error", border_style="red"))
        sys.exit(1)


@main.command()
@click.argument("session_id")
@click.option("--output", "-o", help="Output file path (default: <session_id>.json).")
def export(session_id: str, output: Optional[str]) -> None:
    """Export a session to JSON."""
    try:
        storage = _get_storage()
        session = storage.get_session(session_id)
        if session is None:
            console.print(f"[red]Session '{session_id}' not found.[/red]")
            storage.close()
            sys.exit(1)

        executions = storage.get_node_executions(session_id)
        storage.close()

        data = {
            "session": session.to_dict(),
            "executions": [e.to_dict() for e in executions],
        }

        output_path = output or f"{session_id}.json"
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        console.print(f"[green]Exported to {output_path}[/green]")
    except SystemExit:
        raise
    except Exception as e:
        console.print(Panel(f"[red]Error: {e}[/red]", title="Error", border_style="red"))
        sys.exit(1)


@main.command()
@click.option("--limit", default=50, help="Number of recent runs to analyze.")
@click.option("--provider", default=None, help="Filter by provider name.")
def providers(limit: int, provider: Optional[str]) -> None:
    """Show provider performance leaderboard."""
    try:
        storage = _get_storage()
        summaries = storage.get_provider_leaderboard(limit=limit)
        storage.close()

        if not summaries:
            console.print("[yellow]No provider data yet. Run some agents first.[/yellow]")
            console.print("[dim]Provider stats are collected automatically during recording.[/dim]")
            return

        table = Table(title=f"Provider Leaderboard -- last {limit} runs")
        table.add_column("Provider + Model")
        table.add_column("Avg Latency", justify="right")
        table.add_column("P95 Latency", justify="right")
        table.add_column("Avg Quality", justify="right")
        table.add_column("Total Cost", justify="right")
        table.add_column("Runs", justify="right")
        table.add_column("Badge")

        for s in summaries:
            row_style = ""
            badge = ""
            if s.recommendation == "best_latency":
                row_style = "green"
                badge = "[green]FAST[/green]"
            elif s.recommendation == "best_quality":
                row_style = "blue"
                badge = "[blue]QUALITY[/blue]"
            elif s.recommendation == "best_value":
                row_style = "yellow"
                badge = "[yellow]VALUE[/yellow]"

            quality_str = f"{s.avg_quality_score:.4f}" if s.avg_quality_score is not None else "-"
            table.add_row(
                f"{s.provider}/{s.model}",
                f"{s.avg_latency_ms:.1f}ms",
                f"{s.p95_latency_ms:.1f}ms",
                quality_str,
                f"${s.total_cost_usd:.6f}",
                str(s.run_count),
                badge,
                style=row_style,
            )

        console.print(table)

        # Recommendations panel
        lines = []
        fastest = next((s for s in summaries if s.recommendation == "best_latency"), None)
        if fastest:
            lines.append(f"Best for speed: {fastest.provider}/{fastest.model} ({fastest.avg_latency_ms:.1f}ms avg)")
        best_q = next((s for s in summaries if s.recommendation == "best_quality"), None)
        if best_q:
            lines.append(f"Best for quality: {best_q.provider}/{best_q.model} (score: {best_q.avg_quality_score:.4f})")
        best_v = next((s for s in summaries if s.recommendation == "best_value"), None)
        if best_v:
            per_run = best_v.total_cost_usd / best_v.run_count if best_v.run_count else 0
            lines.append(f"Best value: {best_v.provider}/{best_v.model} (${per_run:.6f} per run)")

        if lines:
            console.print(Panel("\n".join(lines), title="Recommendations", border_style="cyan"))
    except Exception as e:
        console.print(Panel(f"[red]Error: {e}[/red]", title="Error", border_style="red"))
        sys.exit(1)


@main.command()
@click.argument("session_id")
@click.confirmation_option(prompt="Delete session? This cannot be undone.")
def delete(session_id: str) -> None:
    """Delete a session and all its node executions."""
    try:
        storage = _get_storage()
        deleted = storage.delete_session(session_id)
        storage.close()

        if deleted:
            console.print(f"[green]Deleted session '{session_id}'.[/green]")
        else:
            console.print(f"[yellow]Session '{session_id}' not found.[/yellow]")
    except Exception as e:
        console.print(Panel(f"[red]Error: {e}[/red]", title="Error", border_style="red"))
        sys.exit(1)
