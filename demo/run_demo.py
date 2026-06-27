"""End-to-end demo: langgraph-replay + pytest-llm with real LLM calls."""

import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

load_dotenv()

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

if not os.getenv("GROQ_API_KEY"):
    print("[ERROR] GROQ_API_KEY not set in .env")
    sys.exit(1)

from langgraph_replay import BlameEngine, record_session
from research_agent import build_graph, fetch_context, summarize, fact_check, format_output

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

    # ── Step 3b: Auto-diagnosis ──────────────────────────────────────
    console.print(Panel("Step 3b -- Auto-diagnosis", style="bold blue"))
    os.environ["LLM_JUDGE_PROVIDER"] = "mistral"
    os.environ["LLM_JUDGE_MODEL"] = "mistral-small-latest"
    engine_diag = BlameEngine(
        bad_id,
        graph_nodes={
            "fetch_context": fetch_context,
            "summarize": summarize,
            "fact_check": fact_check,
            "format_output": format_output,
        },
    )
    diag_result = engine_diag.run(diagnose=True)
    if diag_result.diagnosis and diag_result.diagnosis.root_cause != "Diagnosis unavailable":
        diag = diag_result.diagnosis
        console.print(Panel(diag.root_cause, title="Why it broke", style="yellow"))
        for i, fix in enumerate(diag.fix_suggestions, 1):
            console.print(f"[green][{i}][/green] {fix}")
    else:
        if diag_result.diagnosis:
            console.print(f"[red]Root cause: {diag_result.diagnosis.root_cause}[/red]")
        else:
            console.print("[red]Diagnosis failed - check GROQ_API_KEY[/red]")

    # ── Step 4: Semantic blame with pytest-llm ────────────────────────
    os.environ["LLM_JUDGE_PROVIDER"] = "mistral"
    os.environ["LLM_JUDGE_MODEL"] = "mistral-small-latest"
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

    # ── Step 6: Provider leaderboard ──────────────────────────────────
    console.print(Panel("Step 6 -- Provider leaderboard", style="bold blue"))
    from langgraph_replay.storage import ReplayStorage
    storage = ReplayStorage()
    leaderboard = storage.get_provider_leaderboard(limit=50)
    storage.close()

    if not leaderboard:
        console.print("[yellow]No provider data yet -- "
                      "run more agents to populate.[/yellow]")
    else:
        table = Table(title="Provider Leaderboard")
        table.add_column("Provider/Model", min_width=30)
        table.add_column("Avg Latency")
        table.add_column("Avg Quality")
        table.add_column("Total Cost")
        table.add_column("Runs")
        table.add_column("Badge")

        for entry in leaderboard:
            badge = entry.recommendation.replace("_", " ").upper() \
                    if entry.recommendation else ""
            color = "green" if entry.recommendation == "best_latency" \
                    else "blue" if entry.recommendation == "best_quality" \
                    else "yellow" if entry.recommendation == "best_value" \
                    else "white"
            table.add_row(
                f"{entry.provider}/{entry.model}",
                f"{entry.avg_latency_ms:.0f}ms",
                f"{entry.avg_quality_score:.2f}"
                    if entry.avg_quality_score else "N/A",
                f"${entry.total_cost_usd:.4f}",
                str(entry.run_count),
                f"[{color}]{badge}[/{color}]",
                style=color if entry.recommendation else ""
            )
        console.print(table)


if __name__ == "__main__":
    main()
