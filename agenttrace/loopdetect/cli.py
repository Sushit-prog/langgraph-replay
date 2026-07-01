"""CLI command for loop detection."""

import sys
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agenttrace.loopdetect.classifier import classify_cycle
from agenttrace.loopdetect.cycle_finder import find_cycles
from agenttrace.loopdetect.embeddings import clear_cache
from langgraph_replay.storage import ReplayStorage

console = Console()


@click.command()
@click.argument("run_id")
@click.option("--threshold", "-t", default=0.92, type=float, help="Similarity threshold for stuck-loop detection (default: 0.92).")
@click.option("--window", "-w", default=3, type=int, help="Compare each visit against previous K occurrences (default: 3).")
@click.option("--output", "-o", default=None, help="Write JSON report to file.")
def loopcheck(run_id: str, threshold: float, window: int, output: Optional[str]):
    """Detect stuck loops in a recorded run.

    Classifies repeated node visits as 'stuck_loop' or 'legitimate_retry'
    based on state similarity and progress heuristics.
    """
    storage = ReplayStorage()
    try:
        executions = storage.get_node_executions(run_id)
        if not executions:
            console.print(f"[yellow]No executions found for run {run_id}.[/yellow]")
            storage.close()
            sys.exit(2)

        cycles = find_cycles(executions)

        if not cycles:
            console.print(f"[green]No repeated node visits detected in run {run_id}.[/green]")
            storage.close()
            sys.exit(0)

        console.print(f"\n[bold]Loop check: run {run_id}[/bold]\n")

        results = []
        for cycle in cycles:
            # Clear embedding cache between cycles to manage memory
            clear_cache()
            classification = classify_cycle(cycle, threshold=threshold, window=window)
            results.append(classification)

            color = "red" if classification.classification == "stuck_loop" else "green"
            icon = "X" if classification.classification == "stuck_loop" else "~"

            console.print(
                f"[bold]Node: {classification.node_name}[/bold] "
                f"(visited {classification.visit_count} times)"
            )
            console.print(
                f"  Classification: [{color}]{classification.classification}[/{color}]"
            )
            console.print(f"  Reasoning: {classification.reasoning}")
            console.print()

        # Summary
        stuck = [r for r in results if r.classification == "stuck_loop"]
        if stuck:
            console.print(
                f"[bold red]Found {len(stuck)} stuck loop(s) in run {run_id}.[/bold red]"
            )
        else:
            console.print(
                f"[bold green]No stuck loops detected in run {run_id}.[/bold green]"
            )

        # Write JSON report if requested
        if output:
            import json
            report = {
                "run_id": run_id,
                "threshold": threshold,
                "window": window,
                "total_cycles": len(results),
                "stuck_loops": sum(1 for r in results if r.classification == "stuck_loop"),
                "cycles": [
                    {
                        "node_name": r.node_name,
                        "visit_count": r.visit_count,
                        "classification": r.classification,
                        "reasoning": r.reasoning,
                        "similarity_scores": r.similarity_scores,
                        "avg_similarity": r.avg_similarity,
                    }
                    for r in results
                ],
            }
            with open(output, "w") as f:
                json.dump(report, f, indent=2)
            console.print(f"\n[dim]JSON report written to {output}[/dim]")

    finally:
        storage.close()
