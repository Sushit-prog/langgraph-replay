"""SQLite storage layer for LangGraph replay sessions."""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class SafeEncoder(json.JSONEncoder):
    """JSON encoder that handles non-serializable types by converting to str."""

    def default(self, obj: Any) -> Any:
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def _serialize_state(state: Any) -> str:
    """Serialize state to JSON, falling back to str for non-serializable types."""
    if state is None:
        return "{}"
    try:
        return json.dumps(state, cls=SafeEncoder)
    except Exception:
        return json.dumps({"_raw": str(state)})


def _deserialize_state(raw: str) -> dict:
    """Deserialize state from JSON string."""
    if not raw:
        return {}
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
        return {"_value": result}
    except json.JSONDecodeError:
        return {"_raw": raw}


class Session(BaseModel):
    """Represents a recorded LangGraph session."""

    id: str
    agent_name: str
    created_at: str
    total_nodes: int = 0
    status: str = "completed"
    final_output: str = ""
    metadata: dict = Field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "agent_name": self.agent_name,
            "created_at": self.created_at,
            "total_nodes": self.total_nodes,
            "status": self.status,
            "final_output": self.final_output,
            "metadata": self.metadata,
        }


class NodeExecution(BaseModel):
    """Represents a single node execution within a session."""

    id: Optional[int] = None
    session_id: str
    node_name: str
    execution_order: int
    input_state: str = ""
    output_state: str = ""
    started_at: str = ""
    duration_ms: float = 0.0
    status: str = "success"
    error_message: Optional[str] = None
    llm_calls: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "node_name": self.node_name,
            "execution_order": self.execution_order,
            "input_state": self.input_state,
            "output_state": self.output_state,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error_message": self.error_message,
            "llm_calls": self.llm_calls,
        }


class ProviderStat(BaseModel):
    """Represents a single LLM provider call statistics."""

    id: Optional[int] = None
    session_id: str
    node_name: str
    provider: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    quality_score: Optional[float] = None
    recorded_at: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "node_name": self.node_name,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "quality_score": self.quality_score,
            "recorded_at": self.recorded_at,
        }


class ProviderSummary(BaseModel):
    """Aggregated provider statistics for leaderboard."""

    provider: str
    model: str
    avg_latency_ms: float
    avg_quality_score: Optional[float] = None
    total_cost_usd: float
    run_count: int
    p95_latency_ms: float
    recommendation: str = ""


class ReplayStorage:
    """SQLite-backed storage for LangGraph replay sessions and node executions."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        agent_name TEXT,
        created_at TEXT,
        total_nodes INT,
        status TEXT,
        final_output TEXT,
        metadata TEXT
    );

    CREATE TABLE IF NOT EXISTS node_executions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        node_name TEXT,
        execution_order INT,
        input_state TEXT,
        output_state TEXT,
        started_at TEXT,
        duration_ms FLOAT,
        status TEXT,
        error_message TEXT,
        llm_calls INT,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    );

    CREATE TABLE IF NOT EXISTS provider_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        node_name TEXT,
        provider TEXT,
        model TEXT,
        latency_ms FLOAT,
        input_tokens INT,
        output_tokens INT,
        cost_usd FLOAT,
        quality_score FLOAT,
        recorded_at TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    );
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize storage with optional custom DB path.

        Args:
            db_path: Path to SQLite database file. Defaults to
                     ~/.langgraph_replay/replays.db or LANGGRAPH_REPLAY_DB env var.
        """
        if db_path is None:
            db_path = os.environ.get("LANGGRAPH_REPLAY_DB")
        if db_path is None:
            db_path = os.path.expanduser("~/.langgraph_replay/replays.db")

        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.SCHEMA)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.commit()

    def save_session(self, session: Session) -> str:
        """Save a session record.

        Args:
            session: Session object to save.

        Returns:
            The session ID.
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions (id, agent_name, created_at, total_nodes, status, final_output, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                session.id,
                session.agent_name,
                session.created_at,
                session.total_nodes,
                session.status,
                session.final_output,
                json.dumps(session.metadata, cls=SafeEncoder),
            ),
        )
        self._conn.commit()
        return session.id

    def save_node_execution(self, execution: NodeExecution) -> None:
        """Save a node execution record.

        Args:
            execution: NodeExecution object to save.
        """
        self._conn.execute(
            "INSERT INTO node_executions (session_id, node_name, execution_order, input_state, output_state, started_at, duration_ms, status, error_message, llm_calls) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                execution.session_id,
                execution.node_name,
                execution.execution_order,
                execution.input_state,
                execution.output_state,
                execution.started_at,
                execution.duration_ms,
                execution.status,
                execution.error_message,
                execution.llm_calls,
            ),
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by ID.

        Args:
            session_id: The session ID to look up.

        Returns:
            Session object or None if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return Session(
            id=row["id"],
            agent_name=row["agent_name"],
            created_at=row["created_at"],
            total_nodes=row["total_nodes"],
            status=row["status"],
            final_output=row["final_output"] or "",
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def get_node_executions(self, session_id: str) -> list[NodeExecution]:
        """Retrieve all node executions for a session, ordered by execution_order.

        Args:
            session_id: The session ID to look up.

        Returns:
            List of NodeExecution objects.
        """
        rows = self._conn.execute(
            "SELECT * FROM node_executions WHERE session_id = ? ORDER BY execution_order",
            (session_id,),
        ).fetchall()
        return [
            NodeExecution(
                id=row["id"],
                session_id=row["session_id"],
                node_name=row["node_name"],
                execution_order=row["execution_order"],
                input_state=row["input_state"] or "",
                output_state=row["output_state"] or "",
                started_at=row["started_at"] or "",
                duration_ms=row["duration_ms"] or 0.0,
                status=row["status"],
                error_message=row["error_message"],
                llm_calls=row["llm_calls"] or 0,
            )
            for row in rows
        ]

    def list_sessions(self, limit: int = 20) -> list[Session]:
        """List most recent sessions, newest first.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of Session objects.
        """
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            Session(
                id=row["id"],
                agent_name=row["agent_name"],
                created_at=row["created_at"],
                total_nodes=row["total_nodes"],
                status=row["status"],
                final_output=row["final_output"] or "",
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in rows
        ]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its node executions.

        Args:
            session_id: The session ID to delete.

        Returns:
            True if found and deleted, False otherwise.
        """
        cursor = self._conn.execute(
            "DELETE FROM node_executions WHERE session_id = ?", (session_id,)
        )
        self._conn.execute(
            "DELETE FROM sessions WHERE id = ?", (session_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0 or self._conn.total_changes > 0

    def search_sessions(
        self, agent_name: Optional[str] = None, status: Optional[str] = None
    ) -> list[Session]:
        """Filter sessions by agent_name and/or status.

        Args:
            agent_name: Filter by agent name.
            status: Filter by session status.

        Returns:
            List of matching Session objects.
        """
        query = "SELECT * FROM sessions WHERE 1=1"
        params: list[Any] = []
        if agent_name:
            query += " AND agent_name = ?"
            params.append(agent_name)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC"

        rows = self._conn.execute(query, params).fetchall()
        return [
            Session(
                id=row["id"],
                agent_name=row["agent_name"],
                created_at=row["created_at"],
                total_nodes=row["total_nodes"],
                status=row["status"],
                final_output=row["final_output"] or "",
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in rows
        ]

    def save_provider_stat(self, stat: ProviderStat) -> None:
        """Save one provider stat record.

        Args:
            stat: ProviderStat object to save.
        """
        self._conn.execute(
            "INSERT INTO provider_stats (session_id, node_name, provider, model, latency_ms, input_tokens, output_tokens, cost_usd, quality_score, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                stat.session_id,
                stat.node_name,
                stat.provider,
                stat.model,
                stat.latency_ms,
                stat.input_tokens,
                stat.output_tokens,
                stat.cost_usd,
                stat.quality_score,
                stat.recorded_at,
            ),
        )
        self._conn.commit()

    def get_provider_stats(
        self, provider: Optional[str] = None, limit: int = 50
    ) -> list[ProviderStat]:
        """Return provider stats, optionally filtered by provider.

        Args:
            provider: Filter by provider name.
            limit: Max rows to return.

        Returns:
            List of ProviderStat objects ordered by recorded_at desc.
        """
        query = "SELECT * FROM provider_stats"
        params: list[Any] = []
        if provider:
            query += " WHERE provider = ?"
            params.append(provider)
        query += " ORDER BY recorded_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [
            ProviderStat(
                id=row["id"],
                session_id=row["session_id"],
                node_name=row["node_name"],
                provider=row["provider"],
                model=row["model"],
                latency_ms=row["latency_ms"],
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                cost_usd=row["cost_usd"],
                quality_score=row["quality_score"],
                recorded_at=row["recorded_at"] or "",
            )
            for row in rows
        ]

    def get_provider_leaderboard(self, limit: int = 50) -> list[ProviderSummary]:
        """Aggregate provider stats into a leaderboard.

        Groups by provider + model, computes averages and p95 latency.

        Args:
            limit: Max rows to return.

        Returns:
            List of ProviderSummary ordered by avg_latency_ms ascending.
        """
        rows = self._conn.execute(
            """SELECT provider, model,
                      AVG(latency_ms) as avg_latency,
                      AVG(quality_score) as avg_quality,
                      SUM(cost_usd) as total_cost,
                      COUNT(*) as run_count
               FROM provider_stats
               GROUP BY provider, model
               ORDER BY avg_latency ASC"""
        ).fetchall()

        summaries: list[ProviderSummary] = []
        for row in rows:
            # Calculate p95 latency for this provider/model
            p95_row = self._conn.execute(
                """SELECT latency_ms FROM provider_stats
                   WHERE provider = ? AND model = ?
                   ORDER BY latency_ms ASC""",
                (row["provider"], row["model"]),
            ).fetchall()
            latencies = [r["latency_ms"] for r in p95_row]
            p95_idx = max(0, int(len(latencies) * 0.95) - 1)
            p95 = latencies[p95_idx] if latencies else 0.0

            summaries.append(ProviderSummary(
                provider=row["provider"],
                model=row["model"],
                avg_latency_ms=round(row["avg_latency"], 2),
                avg_quality_score=round(row["avg_quality"], 4) if row["avg_quality"] is not None else None,
                total_cost_usd=round(row["total_cost"], 6),
                run_count=row["run_count"],
                p95_latency_ms=round(p95, 2),
            ))

        # Assign recommendations
        if summaries:
            # best_latency
            fastest = min(summaries, key=lambda s: s.avg_latency_ms)
            fastest.recommendation = "best_latency"

            # best_quality
            with_quality = [s for s in summaries if s.avg_quality_score is not None]
            if with_quality:
                best_q = max(with_quality, key=lambda s: s.avg_quality_score)
                best_q.recommendation = "best_quality"

            # best_value (quality / cost ratio)
            with_value = [s for s in summaries if s.avg_quality_score is not None and s.total_cost_usd > 0]
            if with_value:
                best_v = max(with_value, key=lambda s: s.avg_quality_score / s.total_cost_usd)
                if best_v.recommendation != "best_quality":
                    best_v.recommendation = "best_value"

        return summaries

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
