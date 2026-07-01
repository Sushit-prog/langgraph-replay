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

    # Show diff strategy if semantic mode was used
    if result.diff_strategy == "semantic" and result.semantic_threshold is not None:
        lines.append(f"Diff strategy: semantic (threshold={result.semantic_threshold})")

    lines.append("")

    for finding in result.findings:
        if finding.finding_type == FindingType.UNCHANGED:
            icon = "[green]~[/green]"
            # Show semantic match info if available
            if finding.semantic_note and "similarity=" in finding.semantic_note:
                status = f"unchanged (semantic match, {finding.semantic_note})"
            else:
                status = "unchanged"
        elif finding.finding_type == FindingType.REGRESSION:
            icon = "[red]![/red]"
            if finding.semantic_note:
                status = f"REGRESSION ({finding.semantic_note})"
            else:
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

            # Phase 5: upstream divergence section
            if finding.upstream_divergences:
                changed_divs = [d for d in finding.upstream_divergences if d.changed]
                total_checked = len(finding.upstream_divergences)
                if changed_divs:
                    lines.append(f"\n  Possible upstream causes:")
                    for div in changed_divs:
                        cat_label = f"[{div.category}]"
                        lines.append(f"    {cat_label} node \"{div.node_name}\" (step {div.step_id})")
                        lines.append(f"      field: {div.field_path}")
                        if div.similarity_score is not None:
                            lines.append(f"      similarity: {div.similarity_score:.2f}")
                        # Truncate long values for readability
                        base_val = str(div.baseline_value)[:100] if div.baseline_value is not None else "N/A"
                        new_val = str(div.new_value)[:100] if div.new_value is not None else "N/A"
                        lines.append(f"      baseline: {base_val!r}")
                        lines.append(f"      new:      {new_val!r}")
                        lines.append(f"      note: {div.note}")
                    lines.append(f"\n  {len(changed_divs)} of {total_checked} checked upstream fields diverged.")

        # Show semantic note for unchanged findings that had raw text differences
        if (finding.finding_type == FindingType.UNCHANGED
                and finding.semantic_note
                and "differs" in finding.semantic_note):
            lines.append(f"    note: raw output text differs but judged equivalent")

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
    report = {
        "baseline_run_id": result.baseline_run_id,
        "new_run_id": result.new_run_id,
        "regression_count": result.regression_count,
        "structural_change_count": result.structural_change_count,
        "has_regression": result.has_regression,
        "diff_strategy": result.diff_strategy,
        "findings": [
            {
                "step_id": f.step_id,
                "node_name": f.node_name,
                "judgment": f.judgment,
                "finding_type": f.finding_type.value,
                "baseline_output": f.baseline_output,
                "new_output": f.new_output,
                "annotation_note": f.annotation_note,
                "semantic_note": f.semantic_note,
                "upstream_divergences": [
                    {
                        "step_id": d.step_id,
                        "node_name": d.node_name,
                        "category": d.category,
                        "field_path": d.field_path,
                        "changed": d.changed,
                        "similarity_score": d.similarity_score,
                        "baseline_value": d.baseline_value,
                        "new_value": d.new_value,
                        "note": d.note,
                    }
                    for d in f.upstream_divergences
                ] if f.upstream_divergences else [],
            }
            for f in result.findings
        ],
    }
    if result.semantic_threshold is not None:
        report["semantic_threshold"] = result.semantic_threshold
    return report


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
