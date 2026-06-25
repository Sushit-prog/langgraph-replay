# langgraph-replay

> Record, replay, debug, diff, and blame LangGraph agent executions.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Tests](https://img.shields.io/badge/tests-31_passing-success)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

LangGraph agents often fail in frustrating ways.

The final output is wrong.

A key disappears somewhere in the graph.

An LLM silently degrades output quality.

A downstream node crashes because an upstream node corrupted state.

Most observability tools show **what happened**.

`langgraph-replay` helps answer:

> **Which node caused the failure, and why?**

Record every node execution, inspect state transitions, replay agent runs step-by-step, compare sessions, and automatically identify the node responsible for a failure.

No cloud service.

No hosted backend.

No vendor lock-in.

Just install, record, and debug.

---

## Why langgraph-replay?

Imagine a simple agent:

```text
fetch_context
      ↓
summarize
      ↓
fact_check
      ↓
format_output
```

The final response is wrong.

Which node caused it?

* Was context lost?
* Did a node mutate state incorrectly?
* Did an LLM produce a degraded response?
* Did quality drop even though the structure remained valid?

Without replay tooling, debugging becomes manual guesswork.

`langgraph-replay` turns every LangGraph execution into a searchable, replayable timeline.

---

## Features

### Automatic Recording

Attach a recorder and capture every node execution automatically.

Recorded metadata includes:

* Node name
* Input state
* Output state
* Execution duration
* Status
* Error message
* LLM call count

All sessions are stored locally in SQLite.

```text
~/.langgraph_replay/replays.db
```

---

### Replay Agent Executions

Inspect every node exactly as it executed.

```bash
langgraph-replay show <session_id>
```

View:

* State before execution
* State after execution
* Metadata
* Timing information
* Errors

---

### Interactive TUI Debugger

Launch a terminal debugger powered by Textual.

```bash
langgraph-replay debug <session_id>
```

Features:

* Node explorer
* State inspection
* State diff viewer
* Metadata panel
* Keyboard navigation

```text
┌──────────────┬──────────────────────┬──────────────┐
│ Node List    │ State Diff Viewer    │ Metadata     │
├──────────────┼──────────────────────┼──────────────┤
│ fetch        │ context added        │ duration     │
│ summarize    │ summary generated    │ llm calls    │
│ fact_check   │ claims verified      │ status       │
│ format       │ final output         │ error info   │
└──────────────┴──────────────────────┴──────────────┘
```

Hotkeys:

| Key | Action         |
| --- | -------------- |
| ↑ ↓ | Navigate nodes |
| D   | State diff     |
| B   | Blame overlay  |
| Q   | Quit           |

---

### Session Diffing

Compare two executions node-by-node.

```bash
langgraph-replay diff session_a session_b
```

Quickly identify:

* State changes
* Missing keys
* Diverging outputs
* Different execution paths

---

### JSON Export

Export executions for sharing or analysis.

```bash
langgraph-replay export <session_id>
```

Output:

```json
{
  "session_id": "...",
  "nodes": [...]
}
```

---

## Blame Engine

The core feature of the project.

Instead of only showing traces, `langgraph-replay` attempts to identify the node responsible for a failure.

### Structural Blame

No LLM required.

Pure state analysis.

Tracks state evolution through the graph and identifies the first node that permanently removed information required by the final output.

Example:

```text
fetch_context
    context ✓

summarize
    context ✗

fact_check
    context missing

format_output
    failure
```

Result:

```text
Likely culprit:
summarize
```

---

### Semantic Blame

Structural correctness does not guarantee output quality.

A node can preserve state while still degrading information.

Semantic blame integrates with `pytest-llm` to compare outputs against a known-good baseline.

```bash
langgraph-replay blame \
    bad_session \
    --eval \
    --baseline good_session
```

The system evaluates:

* Content quality
* Semantic drift
* Regression severity
* Information loss

Example:

```text
Node: format_output

Regression score: 0.60
Threshold: 0.75

Explanation:
Output discusses stock market trends instead of
the requested HNSW indexing algorithm.
```

This catches failures that structural analysis cannot detect.

---

## Quick Start

### Installation

```bash
pip install langgraph-replay
```

Enable semantic blame:

```bash
pip install langgraph-replay[eval]
```

---

### Record an Agent Run

```python
from langgraph_replay import record_session
from langgraph_replay import LangGraphRecorder

with record_session("research-agent"):
    graph.invoke(
        {"query": "What is HNSW indexing?"}
    )
```

Every node execution is automatically captured.

---

### Inspect Sessions

List recent executions:

```bash
langgraph-replay list
```

View a session:

```bash
langgraph-replay show <session_id>
```

Launch debugger:

```bash
langgraph-replay debug <session_id>
```

Run blame analysis:

```bash
langgraph-replay blame <session_id>
```

---

## End-to-End Demo

The repository includes a complete LangGraph research agent.

Graph:

```text
fetch_context
      ↓
summarize
      ↓
fact_check
      ↓
format_output
```

Scenario:

1. Record a successful execution
2. Inject a bug into the summarize node
3. Record a failed execution
4. Compare both runs
5. Run structural blame
6. Run semantic blame
7. Identify the root cause automatically

The demo shows the complete debugging workflow from failure detection to root-cause identification.

---

## Architecture

```text
LangGraph Agent
        │
        ▼
LangGraphRecorder
        │
        ▼
SQLite Session Store
        │
        ├─────────────► Replay Engine
        │
        ├─────────────► Diff Engine
        │
        ├─────────────► TUI Debugger
        │
        ▼
Blame Engine
   │
   ├── Structural Analysis
   │
   └── Semantic Analysis
           │
           ▼
       pytest-llm
```

---

## Project Status

Current implementation includes:

* LangGraph callback recorder
* SQLite-backed storage
* Session replay
* Session diffing
* Interactive Textual debugger
* Structural blame engine
* Semantic blame engine
* JSON export
* Rich CLI
* 31 automated tests

---

## The Bigger Vision

Modern agent systems are becoming increasingly complex.

As graphs grow larger, failures become harder to diagnose.

The goal of `langgraph-replay` is to bring software-style debugging to AI agents:

* Record every execution
* Reproduce failures
* Compare runs
* Identify root causes
* Explain failures

without requiring proprietary observability platforms.

---

## License

MIT License.
