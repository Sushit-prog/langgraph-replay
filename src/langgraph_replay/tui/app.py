"""Textual TUI application for debugging LangGraph sessions."""

from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, ListItem, ListView, Static

from langgraph_replay.blame import BlameEngine
from langgraph_replay.diff import compute_state_diff
from langgraph_replay.replay import ReplayEngine
from langgraph_replay.storage import ReplayStorage, _deserialize_state
from langgraph_replay.tui.widgets import NodeInfoPanel, NodeListItem, StateDiffView


class BlameOverlay(ModalScreen[None]):
    """Modal overlay displaying blame analysis results."""

    BINDINGS = [Binding("escape", "dismiss", "Close")]

    def __init__(self, result) -> None:
        """Initialize with blame result.

        Args:
            result: BlameResult to display.
        """
        super().__init__()
        self.result = result

    def compose(self) -> ComposeResult:
        """Build the blame overlay UI."""
        lines = []

        if self.result.blamed_node:
            lines.append(f"[bold red]Blamed Node: {self.result.blamed_node.node_name}[/bold red]")
            lines.append(f"[red]Reason: {self.result.reason}[/red]")
            lines.append(f"Confidence: {self.result.confidence.upper()}")
        else:
            lines.append("[bold green]No issues found[/bold green]")

        lines.append("")
        lines.append("[bold]Node Analysis:[/bold]")

        for analysis in self.result.analysis:
            icon = "[red]✗[/red]" if analysis.issues_found else "[green]✓[/green]"
            blame_marker = " [bold red]← BLAMED[/bold red]" if analysis.is_blamed else ""
            lines.append(f"{icon} {analysis.node_name}{blame_marker}")
            for issue in analysis.issues_found:
                lines.append(f"    [dim]{issue}[/dim]")

        content = "\n".join(lines)
        yield Static(content, id="blame-content")

    def on_mount(self) -> None:
        """Style the overlay on mount."""
        self.query_one("#blame-content").styles.width = "100%"
        self.query_one("#blame-content").styles.height = "100%"
        self.query_one("#blame-content").styles.padding = 1


class DebugApp(App[None]):
    """TUI application for debugging LangGraph sessions."""

    TITLE = "LangGraph Replay Debugger"

    BINDINGS = [
        Binding("up", "move_up", "Navigate Up", show=True),
        Binding("down", "move_down", "Navigate Down", show=True),
        Binding("b", "blame", "Blame Analysis", show=True),
        Binding("d", "diff", "Diff Previous", show=True),
        Binding("q", "quit", "Quit", show=True),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    CSS = """
    Screen {
        layout: horizontal;
    }
    #node-list {
        width: 30%;
        border: solid green;
        padding: 1;
    }
    #state-viewer {
        width: 40%;
        border: solid blue;
        padding: 1;
    }
    #node-info {
        width: 30%;
        border: solid yellow;
        padding: 1;
    }
    #state-header {
        height: 1;
        padding: 0 1;
        background: $accent;
        color: $text;
    }
    #state-diff {
        height: 1fr;
        padding: 1;
        overflow-y: auto;
    }
    #info-content {
        height: 1fr;
        padding: 1;
        overflow-y: auto;
    }
    """

    def __init__(self, session_id: str, storage: Optional[ReplayStorage] = None) -> None:
        """Initialize the debug app.

        Args:
            session_id: The session ID to debug.
            storage: Optional ReplayStorage instance.
        """
        super().__init__()
        self._session_id = session_id
        self._storage = storage or ReplayStorage()
        self._engine = ReplayEngine(session_id, self._storage)
        self._current_index = 0

    def compose(self) -> ComposeResult:
        """Build the three-panel layout."""
        yield Header()

        with Horizontal():
            with Vertical(id="node-list"):
                yield ListView(id="nodes"), id="node-container"

            with Vertical(id="state-viewer"):
                yield Static("State Diff", id="state-header")
                yield StateDiffView(id="state-diff")

            with Vertical(id="node-info"):
                yield NodeInfoPanel(id="info-content")

        yield Footer()

    def on_mount(self) -> None:
        """Populate the node list on mount."""
        list_view = self.query_one("#nodes", ListView)
        for i, exec in enumerate(self._engine.executions):
            list_view.append(NodeListItem(exec, i))

        if self._engine.executions:
            self._update_panels(0)

    def _update_panels(self, index: int) -> None:
        """Update the state viewer and info panels for the given index."""
        if index < 0 or index >= len(self._engine.executions):
            return

        self._current_index = index
        execution = self._engine.executions[index]

        # Update state diff view
        state_viewer = self.query_one("#state-diff", StateDiffView)
        input_state = _deserialize_state(execution.input_state)
        output_state = _deserialize_state(execution.output_state)
        diff = compute_state_diff(input_state, output_state)
        state_viewer.update(state_viewer.render_diff(diff))

        # Update header
        header = self.query_one("#state-header", Static)
        color = "green" if execution.status == "success" else "red"
        header.update(
            f"Node: [bold]{execution.node_name}[/bold] | "
            f"Status: [{color}]{execution.status}[/{color}] | "
            f"{execution.duration_ms:.1f}ms"
        )

        # Update info panel
        info_panel = self.query_one("#info-content", NodeInfoPanel)
        info_panel.update(info_panel.render_info(execution))

    def action_move_up(self) -> None:
        """Move selection up in the node list."""
        if self._current_index > 0:
            self._current_index -= 1
            list_view = self.query_one("#nodes", ListView)
            list_view.index = self._current_index
            self._update_panels(self._current_index)

    def action_move_down(self) -> None:
        """Move selection down in the node list."""
        if self._current_index < len(self._engine.executions) - 1:
            self._current_index += 1
            list_view = self.query_one("#nodes", ListView)
            list_view.index = self._current_index
            self._update_panels(self._current_index)

    def action_blame(self) -> None:
        """Run blame analysis and show overlay."""
        engine = BlameEngine(self._session_id, self._storage)
        result = engine.run()
        self.push_screen(BlameOverlay(result))

    def action_diff(self) -> None:
        """Show diff between current node and previous node's states."""
        if self._current_index <= 0:
            return

        current = self._engine.executions[self._current_index]
        previous = self._engine.executions[self._current_index - 1]

        prev_output = _deserialize_state(previous.output_state)
        curr_input = _deserialize_state(current.input_state)

        diff = compute_state_diff(prev_output, curr_input)
        state_viewer = self.query_one("#state-diff", StateDiffView)
        header = self.query_one("#state-header", Static)
        header.update(
            f"[bold]Diff: {previous.node_name} → {current.node_name}[/bold]"
        )
        state_viewer.update(state_viewer.render_diff(diff))

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()
