# AgentTrace

<p align="center">

**Understand, Debug, Replay and Validate AI Agent Executions**

Replay LangGraph executions, detect regressions, identify stuck reasoning loops,
trace upstream failures, and validate fixes with counterfactual replay.

</p>

---

## Overview

AgentTrace is an observability and debugging toolkit for AI agents built on top of **LangGraph Replay**.

Instead of only recording execution traces, AgentTrace helps answer questions such as:

- Why did my agent suddenly fail?
- Which execution step introduced the regression?
- Was the failure caused by an upstream tool?
- Is this loop a legitimate retry or an infinite reasoning loop?
- Would changing one intermediate value have prevented the failure?

The project combines execution replay, semantic comparison, root-cause analysis, regression detection, and causal experimentation into a single debugging workflow.

---

# Architecture

<p align="center">
<img src="docs/architecture.png" width="100%">
</p>

The system consists of three major layers:

## Core Replay Engine

Responsible for recording, replaying and inspecting agent executions.

- Execution recorder
- Interactive TUI debugger
- SQLite persistence
- Replay engine
- Semantic search
- Session diffing
- Root-cause diagnosis
- Failure attribution

---

## Regression Intelligence

Builds automated regression analysis on top of recorded executions.

Features include:

- Human annotations
- Baseline management
- Regression watchdog
- Semantic output comparison
- Upstream divergence analysis
- Counterfactual replay

---

## Demo & Fixtures

Includes reproducible LangGraph workflows for testing and experimentation.

---

# Features

## Execution Recording

Capture complete LangGraph executions.

- callback ingestion
- state snapshots
- node timings
- execution metadata
- persistent SQLite storage

---

## Interactive Replay

Replay historical executions step-by-step.

- inspect state transitions
- inspect tool calls
- inspect LLM outputs
- reconstruct execution history

---

## Human Annotation Layer

Label execution steps with human judgement.

Supported labels:

- Correct
- Incorrect
- Expected
- Unexpected

Annotations become ground truth for future regression analysis.

---

## Regression Watchdog

Compare new executions against trusted baselines.

Detects:

- changed outputs
- execution regressions
- missing nodes
- unexpected execution paths

Designed for CI pipelines.

```
Exit Code 0 -> Clean
Exit Code 1 -> Regression
Exit Code 2 -> Configuration Error
```

---

## Semantic State Diffing

Reduce false positives caused by wording differences.

Supports:

- embedding similarity
- configurable thresholds
- exact fallback for structured data

```
Exact Match
        │
        ├── text → semantic similarity
        └── non-text → exact comparison
```

---

## Loop Detection

Automatically classify execution loops.

Detects:

- infinite reasoning loops
- repeated state cycles
- legitimate retries

Uses:

- MiniLM embeddings
- cosine similarity
- configurable window analysis

---

## Upstream Divergence Analysis

When a regression is detected, AgentTrace walks backwards through execution history to identify likely root causes.

Examples include:

- changed tool outputs
- changed retrieved context
- modified documents
- altered execution state

Instead of only reporting:

```
Refund failed.
```

AgentTrace explains:

```
Refund failed

↓

Policy lookup changed

↓

Retriever returned different context

↓

Knowledge base updated
```

---

## Counterfactual Replay

Test causal hypotheses without modifying the original execution.

Example:

```
"What if the policy lookup had returned the old value?"
```

AgentTrace injects baseline values into replayed executions and determines whether the regression disappears.

This enables causal debugging rather than guesswork.

---

# Project Structure

```
agenttrace/

├── annotations/
│   ├── models.py
│   ├── store.py
│   └── cli.py
│
├── watchdog/
│   ├── compare.py
│   ├── baseline.py
│   ├── semantic_diff.py
│   ├── upstream.py
│   ├── report.py
│   └── cli.py
│
├── loopdetect/
│   ├── embeddings.py
│   ├── classifier.py
│   ├── cycle_finder.py
│   └── cli.py
│
├── counterfactual/
│   ├── replay.py
│   └── cli.py
│
├── recorder.py
├── replay.py
├── search.py
├── diagnosis.py
├── diff.py
├── blame.py
├── storage.py
└── cli.py
```

---

# Command Line Interface

## Annotation

```bash
langgraph-replay annotate add <run_id> <step_id> \
    -j correct \
    -n "tool returned valid data"

langgraph-replay annotate list <run_id>

langgraph-replay annotate export <run_id> \
    -o annotations.json
```

---

## Baselines

```bash
langgraph-replay baseline set <run_id>

langgraph-replay baseline show
```

---

## Regression Watchdog

```bash
langgraph-replay watchdog watch <run_id>
```

Semantic comparison

```bash
langgraph-replay watchdog watch \
    <run_id> \
    --semantic \
    --semantic-threshold 0.85
```

Upstream analysis

```bash
langgraph-replay watchdog watch \
    <run_id> \
    --semantic \
    --upstream
```

---

## Loop Detection

```bash
langgraph-replay loopcheck \
    <run_id> \
    --threshold 0.92 \
    --window 3
```

---

## Counterfactual Replay

Manual mode

```bash
langgraph-replay counterfactual test \
    <run_id> \
    --baseline <baseline_id> \
    --graph "my_module:build_graph" \
    --thread-id <thread> \
    --step 2 \
    --field "tool_calls[0].output"
```

Automatic mode

```bash
langgraph-replay counterfactual test \
    <run_id> \
    --from-divergence upstream_report.json
```

---

# Typical Debugging Workflow

```text
Record Execution
        │
        ▼
Replay Session
        │
        ▼
Annotate Correct Steps
        │
        ▼
Pin Baseline
        │
        ▼
Watch Future Runs
        │
        ▼
Regression Found
        │
        ▼
Semantic Comparison
        │
        ▼
Upstream Divergence Analysis
        │
        ▼
Counterfactual Replay
        │
        ▼
Root Cause Confirmed
```

---

# Testing

The project includes comprehensive automated test coverage.

| Module | Tests |
|---------|-------:|
| Annotation Store | 8 |
| Regression Watchdog | 8 |
| Watchdog CLI | 7 |
| Upstream Divergence | 18 |
| Semantic Diff | 16 |
| Loop Classifier | 6 |
| Cycle Finder | 4 |
| Counterfactual Replay | 12 |
| Phase 7 Integration | 9 |
| Existing Tests | 53 |
| **Total** | **141** |

All tests currently pass.

---

# Roadmap

Completed

- Human annotation layer
- Regression watchdog
- Semantic diffing
- Loop detection
- Upstream divergence analysis
- Counterfactual replay
- Demo fixtures
- CLI integration

Future work

- Multi-agent execution graphs
- Distributed trace visualization
- Web dashboard
- OpenTelemetry integration
- Phoenix/LangSmith exporters
- LLM-assisted root-cause explanations

---

# Why AgentTrace?

Modern AI systems are difficult to debug because failures often originate several execution steps before the visible error.

AgentTrace shifts debugging from:

> "The final answer is wrong."

to

> "The retrieval changed, which altered the tool input, which caused the policy lookup to fail, and counterfactual replay confirms this was the root cause."

This makes debugging AI agents reproducible, explainable, and suitable for production workflows.

---

# License

MIT License
