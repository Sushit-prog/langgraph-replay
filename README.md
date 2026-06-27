# langgraph-replay

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-60_passing-success)]()

Record, replay, debug, diff, and blame LangGraph agent executions — all from your terminal.

---

Your LangGraph agent failed. Which node caused it? Why?
How do you fix it?

Langfuse shows you flat spans. LangSmith shows you a tree.
Neither tells you which node dropped a critical state key,
why the output quality degraded, or what to change in your
code to fix it.

langgraph-replay does. It tells you **why it broke** with
auto-diagnosis, shows **exact line numbers** from your source
code in fix suggestions, watches for regressions when you
edit files, and lets you **semantically search** across all
recorded sessions.

```
# Your 6-node agent produced wrong output
# langgraph-replay blame --diagnose session_abc
# → Blamed: summarize (confidence: high)
# → Why it broke: Key 'context' dropped and never recovered
# → Fix: Remove the has_bug conditional at line 5-10
```

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
langgraph-replay blame <id> --diagnose         # + why it broke
langgraph-replay search "summarize failed"     # semantic search
langgraph-replay debug <session_id>            # TUI debugger
```

---

## Architecture

```
LangGraph Agent
       │
       ▼
LangGraphRecorder (LangChain callback)
       │ captures state before/after every node + LLM stats
       ▼
SQLite (~/.langgraph_replay/replays.db)
       │
       ├── langgraph-replay list / show / diff / delete / export
       ├── langgraph-replay blame <session>        ← structural
       ├── langgraph-replay blame --eval           ← semantic via pytest-llm
       ├── langgraph-replay blame --diagnose       ← LLM root cause analysis
       ├── langgraph-replay search <query>         ← semantic search
       ├── langgraph-replay watch <file>           ← regression watchdog
       └── langgraph-replay debug <session>        ← TUI debugger
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
| `langgraph-replay blame <id> --diagnose` | Root cause analysis with fix suggestions |
| `langgraph-replay blame <id> --eval` | Semantic blame via pytest-llm |
| `langgraph-replay search <query>` | Semantic search across all sessions |
| `langgraph-replay watch <file>` | Watch agent file, auto-detect regressions |
| `langgraph-replay providers` | LLM provider performance leaderboard |
| `langgraph-replay export <id>` | Export session to JSON |
| `langgraph-replay delete <id>` | Delete a session |

---

## Blame + Auto-Diagnosis

### Structural blame (instant, no API call)

```bash
langgraph-replay blame session_abc
```

```
Blamed Node: summarize
Reason: Key 'context' dropped and never recovered
Confidence: HIGH

OK fetch_context
X  summarize ← BLAMED
OK fact_check
OK format_output
```

### Diagnosis with source code references

```bash
langgraph-replay blame session_abc --diagnose
```

```
Why it broke:
The node failed due to a hardcoded fallback in the summarize
function (line 5-10) that triggers when state.get('has_bug')
is True, bypassing the LLM entirely.

How to fix it:
[1] Remove the has_bug conditional check (lines 5-10)
[2] Replace hardcoded fallback with try-catch around LLM call
[3] Validate context key before use
```

### Semantic blame (requires pytest-llm)

```bash
pip install langgraph-replay[eval]
langgraph-replay blame session_abc --eval --baseline session_xyz
```

---

## Regression Watchdog

Watch your agent file for changes. When you edit, it
automatically re-runs saved sessions and alerts on regressions.

```bash
langgraph-replay watch research_agent.py --agent-name research_agent
```

Or use programmatically:

```python
from langgraph_replay import RegressionWatchdog

watchdog = RegressionWatchdog(
    agent_file="research_agent.py",
    agent_name="research_agent",
    rerun_fn=my_custom_rerun,
    sessions=5,
)
watchdog.start()  # blocks until Ctrl+C
```

---

## Semantic Search

Find sessions by meaning, not just IDs.

```bash
langgraph-replay search "sessions where summarize failed"
langgraph-replay search "runs with context errors" --threshold 0.2
```

```python
from langgraph_replay import SessionSearchEngine

engine = SessionSearchEngine()
results = engine.search("agent dropped state key")
for r in results:
    print(f"{r.session_id} (score: {r.score:.3f})")
```

---

## Provider Leaderboard

Track which LLM provider performs best across all runs.

```bash
langgraph-replay providers
```

```
Provider/Model                 | Avg Latency | Total Cost | Runs | Badge
groq/llama-3.3-70b-versatile   | 1172ms      | $0.0524    | 135  | FAST
mistral/mistral-small-latest   | 5234ms      | $0.0154    | 14   | VALUE
```

---

## Recording API

```python
# Direct usage
from langgraph_replay import LangGraphRecorder
recorder = LangGraphRecorder(session_name="my_agent")
result = graph.invoke(state, config={"callbacks": [recorder]})
recorder.finalize()

# Context manager
from langgraph_replay import record_session
with record_session("my_agent") as rec:
    result = graph.invoke(state, config={"callbacks": [rec]})

# Async
from langgraph_replay import arecord_session
async with arecord_session("my_agent") as rec:
    result = await graph.ainvoke(state, config={"callbacks": [rec]})
```

---

## Limitations

- **SQLite**: Fine for development and single-machine use. Not recommended for multi-machine teams or high-concurrency production.
- **Callback API**: Relies on LangChain's callback handler interface. May break if LangGraph changes how nodes are invoked between versions.
- **Nested subgraphs**: Experimental. Subgraph node names may collide with parent graph nodes.

---

## License

MIT License — see [LICENSE](LICENSE).
