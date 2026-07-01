"""CLI commands for the regression watchdog."""

import sys
from typing import Optional

import click
from rich.console import Console

from agenttrace.watchdog.baseline import BaselineStore
from agenttrace.watchdog.compare import compare_runs
from agenttrace.watchdog.report import format_json_report, print_report, write_json_report

console = Console()


@click.group()
def watchdog():
    """Regression watchdog — compare runs and detect behavioral regressions."""
    pass


@watchdog.group()
def baseline():
    """Manage the pinned baseline run for regression checks."""
    pass


@baseline.command("set")
@click.argument("run_id")
def baseline_set(run_id: str):
    """Pin a run as the baseline for future watch checks."""
    store = BaselineStore()
    store.set_baseline(run_id)
    console.print(f"[green]Baseline set to run {run_id}[/green]")


@baseline.command("show")
def baseline_show():
    """Show the currently pinned baseline run."""
    store = BaselineStore()
    run_id = store.get_baseline()
    if run_id is None:
        console.print("[yellow]No baseline pinned. Use 'baseline set <run_id>' to pin one.[/yellow]")
        sys.exit(2)
    console.print(f"Baseline run: {run_id}")


@watchdog.command()
@click.argument("new_run_id")
@click.option("--baseline", default=None, help="Baseline run ID. If omitted, uses the pinned baseline.")
@click.option("--output", "-o", default="watchdog-report.json", help="JSON report output path.")
@click.option("--quiet", is_flag=True, help="Suppress human-readable report (JSON + exit code only).")
@click.option("--semantic", is_flag=True, help="Use semantic similarity instead of exact match for output comparison.")
@click.option("--semantic-threshold", default=0.90, type=float, help="Similarity threshold for semantic matching (default: 0.90).")
@click.option("--upstream", is_flag=True, help="Analyze upstream steps for divergent tool outputs on regressions.")
def watch(new_run_id: str, baseline: Optional[str], output: str, quiet: bool, semantic: bool, semantic_threshold: float, upstream: bool):
    """Compare a new run against the baseline and detect regressions.

    Exit codes: 0 = clean, 1 = regression detected, 2 = usage/config error.
    """
    # Resolve baseline
    baseline_id = baseline
    if baseline_id is None:
        bs = BaselineStore()
        baseline_id = bs.get_baseline()
        if baseline_id is None:
            console.print(
                "[yellow]No baseline pinned and no --baseline provided. "
                "Run 'agenttrace baseline set <run_id>' or pass --baseline.[/yellow]"
            )
            sys.exit(2)

    # Run comparison
    try:
        result = compare_runs(
            baseline_run_id=baseline_id,
            new_run_id=new_run_id,
            diff_strategy="semantic" if semantic else "exact",
            semantic_threshold=semantic_threshold,
            include_upstream_divergence=upstream,
        )
    except ValueError as e:
        console.print(f"[yellow]{e}[/yellow]")
        sys.exit(2)

    # Print human-readable report
    print_report(result, quiet=quiet)

    # Write JSON report
    write_json_report(result, output)
    if not quiet:
        console.print(f"\n[dim]JSON report written to {output}[/dim]")

    # Exit code
    if result.has_regression:
        sys.exit(1)
    else:
        sys.exit(0)
