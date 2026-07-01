"""CLI commands for span annotations."""

import getpass
import json
import sys
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from agenttrace.annotations.models import Annotation, Judgment
from agenttrace.annotations.store import AnnotationStore

console = Console()


def _get_store() -> AnnotationStore:
    return AnnotationStore()


@click.group()
def annotate():
    """Annotate individual steps/spans in recorded runs."""
    pass


@annotate.command()
@click.argument("run_id")
@click.argument("step_id")
@click.option("--judgment", "-j", required=True, type=click.Choice(["correct", "incorrect", "expected", "unexpected"]))
@click.option("--note", "-n", default=None, help="Free-form note explaining the judgment.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing annotation for this (run_id, step_id).")
def add(run_id: str, step_id: str, judgment: str, note: Optional[str], overwrite: bool):
    """Annotate a step with a human judgment."""
    store = _get_store()
    try:
        ann = Annotation(
            run_id=run_id,
            step_id=step_id,
            judgment=Judgment(judgment),
            note=note,
            annotator=getpass.getuser(),
        )
        store.save(ann, allow_overwrite=overwrite)
        console.print(f"[green]Annotation saved:[/green] run={run_id} step={step_id} judgment={judgment}")
    except ValueError as e:
        console.print(f"[yellow]{e}[/yellow]")
        store.close()
        sys.exit(1)
    finally:
        store.close()


@annotate.command("list")
@click.argument("run_id")
@click.option("--step", default=None, help="Filter to a specific step.")
def list_annotations(run_id: str, step: Optional[str]):
    """List annotations for a run."""
    store = _get_store()
    try:
        annotations = store.list_by_run(run_id, step_id=step)
        if not annotations:
            console.print(f"[yellow]No annotations found for run {run_id}.[/yellow]")
            return

        table = Table(title=f"Annotations for run {run_id}")
        table.add_column("Step ID", style="cyan")
        table.add_column("Judgment")
        table.add_column("Note")
        table.add_column("Annotated At")

        for ann in annotations:
            style = {
                "correct": "green",
                "incorrect": "red",
                "expected": "blue",
                "unexpected": "yellow",
            }.get(ann.judgment.value, "")
            table.add_row(
                ann.step_id,
                f"[{style}]{ann.judgment.value}[/{style}]",
                ann.note or "-",
                ann.annotated_at[:19],
            )
        console.print(table)
    finally:
        store.close()


@annotate.command()
@click.argument("run_id")
@click.option("--format", "fmt", default="json", type=click.Choice(["json"]), help="Export format.")
@click.option("--output", "-o", default=None, help="Output file path. Default: stdout.")
def export(run_id: str, fmt: str, output: Optional[str]):
    """Export annotations for a run as JSON."""
    store = _get_store()
    try:
        data = store.export(run_id)
        if not data:
            console.print(f"[yellow]No annotations found for run {run_id}.[/yellow]")
            return

        json_str = json.dumps(data, indent=2)
        if output:
            with open(output, "w") as f:
                f.write(json_str)
            console.print(f"[green]Exported to {output}[/green]")
        else:
            console.print(json_str)
    finally:
        store.close()
