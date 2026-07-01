"""CLI command for counterfactual replay testing."""

import json
import sys
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel

from agenttrace.counterfactual.replay import run_counterfactual_test
from agenttrace.watchdog.upstream import UpstreamDivergence

console = Console()


@click.group()
def counterfactual():
    """Counterfactual replay — test causal hypotheses about regressions."""
    pass


@counterfactual.command("test")
@click.argument("run_id")
@click.option("--baseline", required=True, help="Baseline run ID.")
@click.option("--graph", required=True, help="Module path to graph builder (e.g. 'my_agent:build_graph').")
@click.option("--thread-id", required=True, help="Thread ID of the new run to fork from.")
@click.option("--step", required=True, type=int, help="Execution order of the upstream step to test.")
@click.option("--field", required=True, help="Field path from Phase 5 divergence report (e.g. 'tool_calls[0].output').")
@click.option("--category", required=True, type=click.Choice(["tool_output", "retrieved_context"]), help="Category of the divergence.")
@click.option("--baseline-value", required=True, help="The baseline value to substitute (JSON string).")
@click.option("--new-value", required=True, help="The new run's current value (for display).")
@click.option("--node-name", required=True, help="Node name of the upstream step.")
@click.option("--threshold", default=0.85, type=float, help="Semantic similarity threshold (default: 0.85).")
@click.option("--output", "-o", default=None, help="Write JSON report to file.")
def test_counterfactual(
    run_id: str,
    baseline: str,
    graph: str,
    thread_id: str,
    step: int,
    field: str,
    category: str,
    baseline_value: str,
    new_value: str,
    node_name: str,
    threshold: float,
    output: Optional[str],
):
    """Test if a specific upstream divergence causally explains a regression.

    Requires explicit --step, --field, and --category naming exactly which
    upstream divergence to test (matching a field_path from a prior --upstream report).
    """
    # Parse the values from JSON strings
    try:
        parsed_baseline_value = json.loads(baseline_value)
    except json.JSONDecodeError:
        parsed_baseline_value = baseline_value

    try:
        parsed_new_value = json.loads(new_value)
    except json.JSONDecodeError:
        parsed_new_value = new_value

    # Build the divergence object
    divergence = UpstreamDivergence(
        step_id=str(step),
        node_name=node_name,
        category=category,
        field_path=field,
        changed=True,
        similarity_score=None,
        baseline_value=parsed_baseline_value,
        new_value=parsed_new_value,
        note="User-specified divergence for counterfactual test",
    )

    # Build a mock regression finding for the check
    class MockFinding:
        def __init__(self, node_name, new_output):
            self.node_name = node_name
            self.new_output = new_output

    # We need to get the baseline output from the divergence context
    # For now, use the node_name to identify what we're testing
    regression_finding = MockFinding(
        node_name=node_name,
        new_output=parsed_new_value if isinstance(parsed_new_value, str) else json.dumps(parsed_new_value),
    )

    # Get baseline output - this should be the original good output
    # We'll use the baseline_value as a proxy for what the output should be
    baseline_output = parsed_baseline_value if isinstance(parsed_baseline_value, str) else json.dumps(parsed_baseline_value)

    console.print(f"\n[bold]Testing: does baseline's value at \"{node_name}\" (step {step}) explain the regression?[/bold]\n")

    try:
        result = run_counterfactual_test(
            graph_path=graph,
            new_run_thread_id=thread_id,
            divergence=divergence,
            regression_finding=regression_finding,
            baseline_output=baseline_output,
            semantic_threshold=threshold,
        )
    except (ImportError, AttributeError, ValueError) as e:
        console.print(f"[red]Error loading graph: {e}[/red]")
        sys.exit(2)

    # Print results
    console.print("[bold]Substituted baseline value:[/bold]")
    console.print(f"  {str(parsed_baseline_value)[:200]}")
    console.print(f"\n[dim](replacing new run's value:)[/dim]")
    console.print(f"  {str(parsed_new_value)[:200]}")

    console.print(f"\n[bold]Replaying forward from step {step}...[/bold]")
    console.print(f"Replayed output: {result.replayed_output[:200]!r}")
    console.print(f"Baseline output: {result.baseline_output[:200]!r}")
    console.print(f"Similarity to baseline: {result.similarity_to_baseline:.2f} (threshold: {threshold})")

    if result.regression_resolved:
        console.print(Panel(
            f"[bold green]REGRESSION RESOLVED[/bold green]\n\n"
            f"This upstream divergence is implicated as (at least sufficient to be) the cause.\n\n"
            f"{result.note}",
            title="Result",
            border_style="green",
        ))
    else:
        console.print(Panel(
            f"[bold red]REGRESSION NOT RESOLVED[/bold red]\n\n"
            f"This upstream divergence does not explain the regression.\n\n"
            f"{result.note}",
            title="Result",
            border_style="red",
        ))

    # Write JSON report if requested
    if output:
        report = {
            "run_id": run_id,
            "baseline_run_id": baseline,
            "graph_path": graph,
            "thread_id": thread_id,
            "step": step,
            "field": field,
            "category": category,
            "node_name": node_name,
            "regression_resolved": result.regression_resolved,
            "similarity_to_baseline": result.similarity_to_baseline,
            "replayed_output": result.replayed_output,
            "baseline_output": result.baseline_output,
            "original_new_output": result.original_new_output,
            "note": result.note,
        }
        with open(output, "w") as f:
            json.dump(report, f, indent=2)
        console.print(f"\n[dim]JSON report written to {output}[/dim]")

    sys.exit(0 if result.regression_resolved else 1)
