"""4-node LangGraph research agent using Mistral for real LLM calls."""

import os
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph, END
from rich.console import Console

load_dotenv()

console = Console()


class ResearchState(TypedDict):
    topic: str
    context: str
    summary: str
    key_points: List[str]
    fact_check: str
    final_report: str
    has_bug: bool


def _llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
        api_key=os.environ["MISTRAL_API_KEY"],
    )


def fetch_context(state: ResearchState) -> dict:
    console.print("[fetch_context] running...")
    resp = _llm().invoke(
        f"Provide 3 factual paragraphs about: {state['topic']}"
    )
    return {"context": resp.content}


def summarize(state: ResearchState) -> dict:
    console.print("[summarize] running...")
    if state.get("has_bug"):
        return {
            "summary": (
                "The stock market closed higher today. "
                "Tech stocks led gains. Investors remain optimistic."
            )
        }
    resp = _llm().invoke(
        f"Summarize this in 2-3 sentences: {state['context']}"
    )
    return {"summary": resp.content}


def fact_check(state: ResearchState) -> dict:
    console.print("[fact_check] running...")
    resp = _llm().invoke(
        "Fact check this summary against the context. "
        "Reply with VERIFIED or ISSUES FOUND and explanation.\n"
        f"Context: {state['context']}\n"
        f"Summary: {state['summary']}"
    )
    return {"fact_check": resp.content}


def format_output(state: ResearchState) -> dict:
    console.print("[format_output] running...")
    resp = _llm().invoke(
        "Format this into a clean research report:\n"
        f"Topic: {state['topic']}\n"
        f"Summary: {state['summary']}\n"
        f"Fact Check: {state['fact_check']}"
    )
    return {"final_report": resp.content}


def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("fetch_context", fetch_context)
    graph.add_node("summarize", summarize)
    graph.add_node("fact_check", fact_check)
    graph.add_node("format_output", format_output)
    graph.set_entry_point("fetch_context")
    graph.add_edge("fetch_context", "summarize")
    graph.add_edge("summarize", "fact_check")
    graph.add_edge("fact_check", "format_output")
    graph.add_edge("format_output", END)
    return graph.compile()
