# langgraph-replay

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-44_passing-success)]()

Record, replay, debug, diff, and blame LangGraph agent executions — all from your terminal.

---

Your LangGraph agent failed. Which node caused it? Why?
How do you fix it?

Langfuse shows you flat spans. LangSmith shows you a tree.
Neither tells you which node dropped a critical state key,
why the output quality degraded, or what to change in your
code to fix it.

langgraph-replay does.

```
# Your 6-node agent produced wrong output
# Which node caused it?
# Langfuse: shows flat spans — no answer
# LangSmith: shows a tree — no answer
# langgraph-replay:
#   langgraph-replay blame session_abc
#   → Blamed: summarize (confidence: high)
#   → Key 'context' dropped and never recovered
```

---

## Demo

![langgraph-replay demo](assets/demo.gif)

---

## Quick Start

```bash
pip install langgraph-replay
```

```python
from langgraph_replay import record_session

with record_session("my_agent") as rec:
    result = graph.invoke(state, config={"callbacks": [rec]})
```

```bash
langgraph-replay list                          # see sessions
langgraph-replay blame <session_id>            # find the culprit
langgraph-replay debug <session_id>            # TUI debugger
```

---

## Architecture

```
LangGraph Agent
       │
       ▼
LangGraphRecorder (LangChain callback)
       │ captures state before/after every node
       ▼
SQLite (~/.langgraph_replay/replays.db)
       │
       ├── langgraph-replay list
       ├── langgraph-replay show <session>
       ├── langgraph-replay diff <a> <b>
       ├── langgraph-replay blame <session>     ← structural
       ├── langgraph-replay blame --eval        ← semantic via pytest-llm
       └── langgraph-replay debug <session>     ← TUI debugger
```

---

## Recording API

### Direct usage

```python
from langgraph_replay import LangGraphRecorder

recorder = LangGraphRecorder(session_name="research_agent")
result = graph.invoke(state, config={"callbacks": [recorder]})
session_id = recorder.finalize()
```

### Context manager

```python
from langgraph_replay import record_session

with record_session("research_agent") as rec:
    result = graph.invoke(state, config={"callbacks": [rec]})
# Session auto-saved on exit
```

### Async support

```python
from langgraph_replay import arecord_session

async with arecord_session("my_agent") as rec:
    result = await graph.ainvoke(state, config={"callbacks": [rec]})
print(rec.session_id)
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `langgraph-replay list` | List recent recorded sessions |
| `langgraph-replay show <id>` | Show session details and node executions |
| `langgraph-replay debug <id>` | Launch TUI debugger |
| `langgraph-replay diff <a> <b>` | Compare two sessions side-by-side |
| `langgraph-replay blame <id>` | Identify which node caused a failure |
| `langgraph-replay export <id>` | Export session to JSON |
| `langgraph-replay delete <id>` | Delete a session |

---

## Blame Modes

### Structural (instant, no API call)

Pure state analysis. Tracks keys through the graph and finds the first node that permanently dropped information required by the final output.

```bash
langgraph-replay blame session_abc
```

```
Blame Analysis
Blamed Node: summarize
Reason: Key 'context' dropped and never recovered
Confidence: HIGH

OK fetch_context
X  summarize ← BLAMED
OK fact_check
OK format_output
```

### Semantic (requires `pytest-llm`)

Compares text outputs against a known-good baseline using `assert_regression`. Catches quality degradation that structural analysis misses.

```bash
pip install langgraph-replay[eval]
langgraph-replay blame session_abc --eval --baseline session_xyz
```

If no `--baseline` is provided, the most recent completed session for the same agent is used automatically.

```
Blame Analysis
Blamed Node: format_output
Reason: Semantic regression on key 'answer': similarity 0.62 < threshold
Confidence: HIGH
```

---

## pytest-llm Integration

```bash
pip install langgraph-replay[eval]
```

```bash
langgraph-replay blame session_abc --eval
langgraph-replay blame session_abc --eval --baseline session_xyz
```

Requires a [pytest-llm](https://github.com/pytest-dev/pytest-llm) compatible setup. The `--eval` flag enables semantic comparison of text outputs across sessions.

---

## Async Support

`graph.invoke()` is fully supported out of the box.

For `graph.ainvoke()`, use the async context manager:

```python
from langgraph_replay import arecord_session

async with arecord_session("my_agent") as rec:
    result = await graph.ainvoke(state, config={"callbacks": [rec]})

print(rec.session_id)
```

The `LangGraphRecorder` callback handler works with both sync and async LangGraph invocations.

---

## Limitations

- **SQLite**: Fine for development and single-machine use. Not recommended for multi-machine teams or high-concurrency production.
- **Callback API**: Relies on LangChain's callback handler interface. May break if LangGraph changes how nodes are invoked between versions.
- **Nested subgraphs**: Experimental. Subgraph node names may collide with parent graph nodes.

---

## License

MIT License — see [LICENSE](LICENSE).
