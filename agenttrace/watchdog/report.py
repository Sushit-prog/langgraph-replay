"""Human-readable + JSON report formatting for watchdog comparison results."""

import json
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from agenttrace.watchdog.compare import ComparisonResult, FindingType

console = Console()


def format_human_readable(result: ComparisonResult) -> str:
    """Format a ComparisonResult as a human-readable report string."""
    lines: list[str] = []

    header = f"Watchdog: comparing run {result.new_run_id} against baseline run {result.baseline_run_id}"
    lines.append(header)
    lines.append("")

    for finding in result.findings:
        if finding.finding_type == FindingType.UNCHANGED:
            icon = "[green]~[/green]"
            status = "unchanged"
        elif finding.finding_type == FindingType.REGRESSION:
            icon = "[red]![/red]"
            status = "REGRESSION"
        else:
            icon = "[yellow]?[/yellow]"
            status = "structural_change (not present in new run)"

        lines.append(f"{icon} {finding.node_name:30s} {status}")

        if finding.finding_type == FindingType.REGRESSION:
            # Truncate long outputs for readability
            base_out = (finding.baseline_output or "")[:120]
            new_out = (finding.new_output or "")[:120]
            lines.append(f"    baseline: {base_out!r}")
            lines.append(f"    new run:  {new_out!r}")

        if finding.annotation_note:
            lines.append(f"    note: {finding.annotation_note}")

    lines.append("")

    if result.has_regression:
        lines.append(
            f"[bold red]Result: {result.regression_count} regression(s) found. Exit code 1.[/bold red]"
        )
    else:
        lines.append("[bold green]Result: No regressions found. Exit code 0.[/bold green]")

    return "\n".join(lines)


def format_json_report(result: ComparisonResult) -> dict:
    """Format a ComparisonResult as a machine-readable JSON-serializable dict.

    Schema is locked for Phase 3/4 consumption — any key changes must be
    coordinated with downstream consumers.
    """
    return {
        "baseline_run_id": result.baseline_run_id,
        "new_run_id": result.new_run_id,
        "regression_count": result.regression_count,
        "structural_change_count": result.structural_change_count,
        "has_regression": result.has_regression,
        "findings": [
            {
                "step_id": f.step_id,
                "node_name": f.node_name,
                "judgment": f.judgment,
                "finding_type": f.finding_type.value,
                "baseline_output": f.baseline_output,
                "new_output": f.new_output,
                "annotation_note": f.annotation_note,
            }
            for f in result.findings
        ],
    }


def print_report(result: ComparisonResult, quiet: bool = False) -> None:
    """Print the human-readable report to the console (unless quiet).

    Always writes JSON to the returned dict for programmatic use.
    """
    if not quiet:
        report_text = format_human_readable(result)
        console.print(report_text)


def write_json_report(result: ComparisonResult, output_path: str) -> None:
    """Write the JSON report to a file."""
    data = format_json_report(result)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
