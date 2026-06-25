# langgraph-replay + pytest-llm Demo

End-to-end demo showing real LLM call recording, structural blame, and semantic blame via pytest-llm.

## What this demonstrates

1. **Good run** — a 4-node research agent produces a coherent report
2. **Bad run** — a simulated bug in the `summarize` node causes the agent to output an unrelated summary
3. **Structural blame** — `BlameEngine` detects key drops and errors without comparing to a baseline
4. **Semantic blame** — `BlameEngine` with `use_eval=True` and a baseline session catches the `summarize` node's output drift via `pytest_llm.assert_regression`
5. **CLI diff** — compare two sessions side by side

## Setup

```bash
# Install the package with eval extras
pip install langgraph-replay[eval]

# Install demo dependencies
pip install langchain-groq python-dotenv

# Create .env with your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env
```

Get a free API key at [console.groq.com](https://console.groq.com).

## Run

```bash
python demo/run_demo.py
```

## What to expect

- **Step 1** prints a coherent research report about HNSW indexing (green panel)
- **Step 2** prints a report with an unrelated stock market summary (red panel)
- **Step 3** structural blame may not find issues since all nodes completed without errors — it only looks for dropped keys and error statuses
- **Step 4** semantic blame compares the bad run's `summarize` output against the good run's baseline, catching the regression and blaming the `summarize` node with high confidence
- **Step 5** prints a `langgraph-replay diff` command you can run manually

## Manual CLI commands

After running the demo, the session IDs are printed. You can also:

```bash
# List all recorded sessions
langgraph-replay list

# Show details for a session
langgraph-replay show <session_id>

# Diff two sessions
langgraph-replay diff <good_session_id> <bad_session_id>

# Run blame analysis
langgraph-replay blame <bad_session_id>

# Run semantic blame with a baseline
langgraph-replay blame <bad_session_id> --baseline <good_session_id> --eval
```
