"""End-to-end demo: langgraph-replay + pytest-llm with real LLM calls."""

import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print("[ERROR] GROQ_API_KEY not set in .env")
    sys.exit(1)

from langgraph_replay import BlameEngine, record_session
from research_agent import build_graph

console = Console()

TOPIC = "HNSW indexing algorithm"


def main():
    graph = build_graph()
    initial = {
        "topic": TOPIC,
        "context": "",
        "summary": "",
        "key_points": [],
        "fact_check": "",
        "final_report": "",
        "has_bug": False,
    }

    # ── Step 1: Good run ──────────────────────────────────────────────
    console.print(Panel("Step 1 — Good run", style="green bold"))
    with record_session("demo_good") as rec:
        result = graph.invoke({**initial, "has_bug": False}, config={"callbacks": [rec]})
    good_id = rec.session_id
    console.print(f"Session ID: {good_id}")
    console.print(
        Panel(result["final_report"][:300] + "...", title="final_report", border_style="green")
    )

    # ── Step 2: Bad run (bug active) ──────────────────────────────────
    console.print(Panel("Step 2 — Bad run (bug active)", style="red bold"))
    with record_session("demo_bad") as rec:
        result = graph.invoke({**initial, "has_bug": True}, config={"callbacks": [rec]})
    bad_id = rec.session_id
    console.print(f"Session ID: {bad_id}")
    console.print(
        Panel(result["final_report"][:300] + "...", title="final_report", border_style="red")
    )

    # ── Step 3: Structural blame ──────────────────────────────────────
    console.print(Panel("Step 3 — Structural blame", style="yellow bold"))
    engine = BlameEngine(bad_id)
    structural = engine.run()
    if structural.blamed_node:
        console.print(f"Blamed: [bold]{structural.blamed_node.node_name}[/bold]")
        console.print(f"Confidence: {structural.confidence}")
        console.print(f"Reason: {structural.reason}")
    else:
        console.print("No structural issues found.")

    # ── Step 4: Semantic blame with pytest-llm ────────────────────────
    os.environ["LLM_JUDGE_PROVIDER"] = "groq"
    os.environ["LLM_JUDGE_MODEL"] = "llama-3.3-70b-versatile"
    console.print(Panel("Step 4 — Semantic blame (pytest-llm)", style="magenta bold"))
    engine = BlameEngine(bad_id)
    semantic = engine.run(baseline_session_id=good_id, use_eval=True)
    if semantic.blamed_node:
        console.print(f"Blamed: [bold]{semantic.blamed_node.node_name}[/bold]")
        console.print(f"Confidence: {semantic.confidence}")
        console.print(f"Reason: {semantic.reason}")
    else:
        console.print("No semantic regressions detected.")

    # ── Step 5: CLI diff hint ─────────────────────────────────────────
    console.print(Panel("Step 5 — Try it yourself", style="cyan bold"))
    console.print("Run a diff between the two sessions:")
    console.print(
        Syntax(
            f"langgraph-replay diff {good_id} {bad_id}",
            "bash",
            theme="monokai",
        )
    )


if __name__ == "__main__":
    main()
