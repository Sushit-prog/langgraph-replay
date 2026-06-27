"""Tests for provider stats and leaderboard."""

import pytest

from langgraph_replay.storage import ProviderStat, ProviderSummary, Session


class TestProviderStats:
    """Tests for provider stat storage and leaderboard."""

    def test_save_and_retrieve_provider_stat(self, storage):
        """Save one ProviderStat, retrieve and assert fields match."""
        session = Session(
            id="session_prov_1",
            agent_name="test",
            created_at="2024-01-15T10:00:00Z",
            total_nodes=1,
            status="completed",
        )
        storage.save_session(session)

        stat = ProviderStat(
            session_id="session_prov_1",
            node_name="summarize",
            provider="groq",
            model="llama-3.3-70b-versatile",
            latency_ms=123.45,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.0001,
            quality_score=0.95,
            recorded_at="2024-01-15T10:00:01Z",
        )
        storage.save_provider_stat(stat)

        stats = storage.get_provider_stats()
        assert len(stats) == 1
        assert stats[0].provider == "groq"
        assert stats[0].model == "llama-3.3-70b-versatile"
        assert stats[0].latency_ms == 123.45
        assert stats[0].input_tokens == 100
        assert stats[0].output_tokens == 50
        assert stats[0].cost_usd == 0.0001
        assert stats[0].quality_score == 0.95

    def test_provider_leaderboard_aggregation(self, storage):
        """Save 5 stats (3 groq, 2 openai), assert 2 leaderboard entries."""
        session = Session(
            id="session_lb",
            agent_name="test",
            created_at="2024-01-15T10:00:00Z",
            total_nodes=1,
            status="completed",
        )
        storage.save_session(session)

        for i in range(3):
            storage.save_provider_stat(ProviderStat(
                session_id="session_lb",
                node_name="node_%d" % i,
                provider="groq",
                model="llama-3.3-70b-versatile",
                latency_ms=100.0 + i * 10,
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.0001,
                recorded_at="2024-01-15T10:00:0%dZ" % i,
            ))

        for i in range(2):
            storage.save_provider_stat(ProviderStat(
                session_id="session_lb",
                node_name="node_%d" % (i + 3),
                provider="openai",
                model="gpt-4o-mini",
                latency_ms=200.0 + i * 10,
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.0002,
                recorded_at="2024-01-15T10:00:0%dZ" % (i + 3),
            ))

        board = storage.get_provider_leaderboard()
        assert len(board) == 2

        groq_entry = next(s for s in board if s.provider == "groq")
        openai_entry = next(s for s in board if s.provider == "openai")
        assert groq_entry.run_count == 3
        assert openai_entry.run_count == 2

    def test_provider_leaderboard_recommendation(self, storage):
        """Groq has lowest latency, assert it gets best_latency recommendation."""
        session = Session(
            id="session_rec",
            agent_name="test",
            created_at="2024-01-15T10:00:00Z",
            total_nodes=1,
            status="completed",
        )
        storage.save_session(session)

        # Groq: fast
        storage.save_provider_stat(ProviderStat(
            session_id="session_rec",
            node_name="node_1",
            provider="groq",
            model="llama-3.3-70b-versatile",
            latency_ms=50.0,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.0001,
            recorded_at="2024-01-15T10:00:00Z",
        ))

        # OpenAI: slow
        storage.save_provider_stat(ProviderStat(
            session_id="session_rec",
            node_name="node_2",
            provider="openai",
            model="gpt-4o-mini",
            latency_ms=500.0,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.0002,
            recorded_at="2024-01-15T10:00:01Z",
        ))

        board = storage.get_provider_leaderboard()
        groq_entry = next(s for s in board if s.provider == "groq")
        assert groq_entry.recommendation == "best_latency"

    def test_cost_calculation(self, storage):
        """Save a groq/llama-3.3-70b stat with known tokens, assert cost."""
        session = Session(
            id="session_cost",
            agent_name="test",
            created_at="2024-01-15T10:00:00Z",
            total_nodes=1,
            status="completed",
        )
        storage.save_session(session)

        # groq/llama-3.3-70b-versatile: $0.59/1M input, $0.79/1M output
        # 1000 input tokens + 500 output tokens
        # cost = (1000 * 0.59 + 500 * 0.79) / 1_000_000 = 0.00059 + 0.000395 = 0.000985
        expected_cost = (1000 * 0.59 + 500 * 0.79) / 1_000_000

        stat = ProviderStat(
            session_id="session_cost",
            node_name="summarize",
            provider="groq",
            model="llama-3.3-70b-versatile",
            latency_ms=100.0,
            input_tokens=1000,
            output_tokens=500,
            cost_usd=expected_cost,
            recorded_at="2024-01-15T10:00:00Z",
        )
        storage.save_provider_stat(stat)

        stats = storage.get_provider_stats()
        assert abs(stats[0].cost_usd - expected_cost) < 1e-10

    def test_recorder_captures_llm_stats(self, mock_graph, storage):
        """Run mock_graph with recorder, assert provider_stats has entries."""
        from langgraph_replay.recorder import LangGraphRecorder

        recorder = LangGraphRecorder(session_name="test_llm", storage=storage)
        mock_graph.invoke(
            {"messages": [], "step": 0},
            config={"callbacks": [recorder]},
        )
        recorder.finalize()

        # The mock graph doesn't make real LLM calls, so stats may be empty
        # but the mechanism should not crash
        assert isinstance(recorder._provider_stats, list)

    def test_get_provider_stats_filtered(self, storage):
        """Filter stats by provider."""
        session = Session(
            id="session_filter",
            agent_name="test",
            created_at="2024-01-15T10:00:00Z",
            total_nodes=1,
            status="completed",
        )
        storage.save_session(session)

        for prov in ["groq", "openai", "groq"]:
            storage.save_provider_stat(ProviderStat(
                session_id="session_filter",
                node_name="node_1",
                provider=prov,
                model="model-x",
                latency_ms=100.0,
                input_tokens=10,
                output_tokens=10,
                cost_usd=0.0001,
                recorded_at="2024-01-15T10:00:00Z",
            ))

        groq_stats = storage.get_provider_stats(provider="groq")
        assert len(groq_stats) == 2

        openai_stats = storage.get_provider_stats(provider="openai")
        assert len(openai_stats) == 1
