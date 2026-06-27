"""Semantic session search using sentence-transformers."""

import json
from typing import Optional

from pydantic import BaseModel

from langgraph_replay.storage import ReplayStorage, Session

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def _get_model():
    """Lazy load sentence-transformers model."""
    global _model
    if _model is None:
        import logging
        logging.getLogger("sentence_transformers").setLevel(
            logging.ERROR
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(MODEL_NAME)
    return _model


def _embed(text: str) -> list[float]:
    """Returns embedding vector for text."""
    model = _get_model()
    return model.encode(text).tolist()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    import numpy as np
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SearchResult(BaseModel):
    """A single search result."""
    session_id: str
    agent_name: str
    status: str
    score: float
    total_nodes: int
    created_at: str
    snippet: str


class SessionSearchEngine:
    """Semantic search over recorded LangGraph sessions.

    Usage::

        engine = SessionSearchEngine()
        results = engine.search(
            "sessions where summarize failed"
        )
        for result in results:
            print(result.session_id, result.score)
    """

    def __init__(self, storage: Optional[ReplayStorage] = None):
        self.storage = storage or ReplayStorage()

    def _session_to_text(self, session_id: str) -> str:
        """Converts a session to searchable text."""
        session = self.storage.get_session(session_id)
        if not session:
            return ""

        executions = self.storage.get_node_executions(session_id)

        node_names = [e.node_name for e in executions]
        failed_nodes = [
            e.node_name for e in executions if e.status == "error"
        ]

        state_keys = set()
        for e in executions:
            try:
                state = json.loads(e.input_state or "{}")
                state_keys.update(state.keys())
            except json.JSONDecodeError:
                pass

        final_output = ""
        if session.final_output:
            try:
                output = json.loads(session.final_output)
                final_output = str(output)[:200]
            except json.JSONDecodeError:
                final_output = str(session.final_output)[:200]

        return (
            f"Agent: {session.agent_name}\n"
            f"Status: {session.status}\n"
            f"Nodes: {' -> '.join(node_names)}\n"
            f"Failed nodes: {', '.join(failed_nodes) or 'none'}\n"
            f"State keys: {', '.join(sorted(state_keys))}\n"
            f"Final output: {final_output}"
        )

    def search(
        self, query: str, limit: int = 5, threshold: float = 0.3
    ) -> list[SearchResult]:
        """Searches all sessions semantically."""
        sessions = self.storage.list_sessions(limit=100)
        if not sessions:
            return []

        query_embedding = _embed(query)
        results = []

        for session in sessions:
            text = self._session_to_text(session.id)
            if not text:
                continue
            session_embedding = _embed(text)
            score = _cosine_similarity(query_embedding, session_embedding)
            if score >= threshold:
                results.append(SearchResult(
                    session_id=session.id,
                    agent_name=session.agent_name,
                    status=session.status,
                    score=score,
                    total_nodes=session.total_nodes,
                    created_at=session.created_at,
                    snippet=text[:150],
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    def find_similar(
        self, session_id: str, limit: int = 5
    ) -> list[SearchResult]:
        """Finds sessions similar to a given session."""
        text = self._session_to_text(session_id)
        if not text:
            return []
        return self.search(text, limit=limit + 1)
