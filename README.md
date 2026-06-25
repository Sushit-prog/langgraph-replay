# langgraph-replay

Record, replay, and debug LangGraph agent executions from your terminal.

## Install

```bash
pip install langgraph-replay
```

## Quick Start

```python
from langgraph_replay import record_session

with record_session("my_agent") as rec:
    result = graph.invoke(state, config={"callbacks": [rec]})

# Debug in terminal
# langgraph-replay debug <session_id>
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `langgraph-replay list` | List recent recorded sessions |
| `langgraph-replay show <id>` | Show session details and node executions |
| `langgraph-replay debug <id>` | Launch TUI debugger |
| `langgraph-replay diff <id_a> <id_b>` | Compare two sessions |
| `langgraph-replay blame <id>` | Identify which node caused a failure |
| `langgraph-replay export <id>` | Export session to JSON |
| `langgraph-replay delete <id>` | Delete a session |

## Recording API

### LangGraphRecorder

```python
from langgraph_replay import LangGraphRecorder

recorder = LangGraphRecorder(
    session_name="research_agent",
    metadata={"user_id": "abc123"}
)

result = graph.invoke(
    initial_state,
    config={"callbacks": [recorder]}
)

session_id = recorder.session_id
```

### record_session Context Manager

```python
from langgraph_replay import record_session

with record_session("my_agent") as rec:
    result = graph.invoke(state, config={"callbacks": [rec]})

# Session is auto-saved when the context exits
```

## Blame Mode

Blame walks backwards through your execution to find which node introduced a failure.

```bash
langgraph-replay blame session_abc123
```

Output:
```
 Blame Analysis

 Blamed Node: process_data
 Reason: Node 'process_data' raised an error: ValueError
 Confidence: HIGH

 ✓ load_data
 ✓ validate_input
 ✗ process_data ← BLAMED
     Node raised error: ValueError
 ✓ save_results
```

## pytest-llm Integration

For semantic comparison between sessions, install the eval extras:

```bash
pip install langgraph-replay[eval]
```

Then use blame with a baseline session:

```bash
langgraph-replay blame session_abc --eval --baseline session_xyz
```

This compares text outputs between sessions using similarity thresholds.

## How It Works

langgraph-replay uses LangChain's callback handler system to intercept node executions. When you pass a `LangGraphRecorder` as a callback, it captures state before and after each node runs, timing, and errors—all stored in a local SQLite database. The TUI debugger loads this data and lets you step through executions, view state diffs, and run blame analysis.
