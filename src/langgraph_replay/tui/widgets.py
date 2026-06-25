"""Custom Textual widgets for the LangGraph replay debugger."""

from textual.widgets import ListItem, Static

from langgraph_replay.diff import StateDiff
from langgraph_replay.storage import NodeExecution, _deserialize_state


class NodeListItem(ListItem):
    """Single item in the node list. Stores NodeExecution reference."""

    def __init__(self, execution: NodeExecution, index: int) -> None:
        """Initialize the node list item.

        Args:
            execution: The NodeExecution to display.
            index: The display index.
        """
        self.execution = execution
        color = "green" if execution.status == "success" else "red"
        duration = f"{execution.duration_ms:.1f}ms"
        label = f"[{index}] [{color}]{execution.node_name}[/{color}] ({duration})"
        super().__init__(Static(label))


class StateDiffView(Static):
    """Renders a StateDiff as colored Rich markup."""

    def render_diff(self, diff: StateDiff) -> str:
        """Render a StateDiff as Rich markup string.

        Args:
            diff: The StateDiff to render.

        Returns:
            Rich markup string for display.
        """
        lines = []

        for key, val in diff.added.items():
            val_str = self._truncate(val)
            lines.append(f"  [green]+ {key}: {val_str}[/green]")

        for key, val in diff.removed.items():
            val_str = self._truncate(val)
            lines.append(f"  [red]- {key}: {val_str}[/red]")

        for key, vals in diff.modified.items():
            before_str = self._truncate(vals.get("before"))
            after_str = self._truncate(vals.get("after"))
            lines.append(f"  [yellow]~ {key}:[/yellow]")
            lines.append(f"    [red]- {before_str}[/red]")
            lines.append(f"    [green]+ {after_str}[/green]")

        for key in diff.unchanged:
            lines.append(f"    {key}: (unchanged)")

        if not lines:
            return "  [dim]No state changes[/dim]"

        return "\n".join(lines)

    def _truncate(self, value: object, max_len: int = 200) -> str:
        """Truncate a value to max_len characters.

        Args:
            value: The value to truncate.
            max_len: Maximum characters.

        Returns:
            Truncated string representation.
        """
        s = str(value)
        if len(s) > max_len:
            return s[:max_len] + "..."
        return s


class NodeInfoPanel(Static):
    """Renders node metadata."""

    def render_info(self, execution: NodeExecution) -> str:
        """Render node execution info as Rich markup.

        Args:
            execution: The NodeExecution to display.

        Returns:
            Rich markup string for display.
        """
        status_color = "green" if execution.status == "success" else "red"
        lines = [
            f"[bold]Node: {execution.node_name}[/bold]",
            f"Order: {execution.execution_order}",
            f"Started: {execution.started_at}",
            f"Duration: {execution.duration_ms:.1f}ms",
            f"Status: [{status_color}]{execution.status}[/{status_color}]",
        ]

        if execution.error_message:
            lines.append(f"[red]Error: {execution.error_message}[/red]")

        if execution.llm_calls:
            lines.append(f"LLM Calls: {execution.llm_calls}")

        input_state = _deserialize_state(execution.input_state)
        if input_state:
            lines.append("")
            lines.append("[bold]Input State:[/bold]")
            for key, val in input_state.items():
                if key.startswith("_"):
                    continue
                val_str = str(val)
                if len(val_str) > 200:
                    val_str = val_str[:200] + "..."
                lines.append(f"  {key}: {val_str}")

        return "\n".join(lines)
