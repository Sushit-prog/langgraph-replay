"""Auto-diagnosis engine: explains WHY a node broke and HOW to fix it."""

import json
import logging
import os
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from langgraph_replay.blame import BlameResult
from langgraph_replay.diff import StateDiff, compute_state_diff
from langgraph_replay.storage import NodeExecution, ReplayStorage, _deserialize_state

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert LangGraph debugger. You analyze node \
execution data from a LangGraph agent and provide:
1. A clear explanation of why the node failed or degraded
2. Specific, actionable steps to fix the issue

Focus on:
- State management issues (dropped keys, wrong types)
- Prompt template problems
- LLM output format issues
- Node function logic errors

Be specific and technical. Assume the developer knows Python \
and LangGraph. Do not be generic.

Respond only in JSON with keys: \
root_cause, fix_suggestions (list of strings), confidence"""

DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
}


class DiagnosisResult(BaseModel):
    """Result of LLM-based root cause analysis."""

    session_id: str
    blamed_node_name: Optional[str] = None
    root_cause: str = "No issues detected"
    fix_suggestions: list[str] = Field(default_factory=list)
    confidence: str = "low"
    raw_response: str = ""


class DiagnosisEngine:
    """Takes a blame result and generates root cause analysis and fix suggestions.

    Usage::

        engine = DiagnosisEngine("session_abc123")
        result = engine.diagnose(blame_result)
        print(result.root_cause)
        for fix in result.fix_suggestions:
            print(f"  - {fix}")
    """

    def __init__(
        self,
        session_id: str,
        storage: Optional[ReplayStorage] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize the diagnosis engine.

        Args:
            session_id: The session ID being diagnosed.
            storage: Optional ReplayStorage instance.
            provider: LLM provider. Reads from LLM_JUDGE_PROVIDER env var,
                      defaults to "groq".
            model: LLM model. Reads from LLM_JUDGE_MODEL env var,
                   defaults per provider.
        """
        self._storage = storage or ReplayStorage()
        self._session_id = session_id
        self._provider = provider or os.environ.get("LLM_JUDGE_PROVIDER", "groq")
        self._model = model or os.environ.get(
            "LLM_JUDGE_MODEL", DEFAULT_MODELS.get(self._provider, "gpt-4o-mini")
        )

    def diagnose(self, blame_result: BlameResult) -> DiagnosisResult:
        """Run diagnosis on a blame result.

        Args:
            blame_result: BlameResult from BlameEngine.run().

        Returns:
            DiagnosisResult with root cause and fix suggestions.
        """
        if blame_result.blamed_node is None:
            return DiagnosisResult(
                session_id=self._session_id,
                root_cause="No issues detected",
                fix_suggestions=[],
                confidence="low",
            )

        blamed = blame_result.blamed_node

        # Compute state diff for the blamed node
        input_state = _deserialize_state(blamed.input_state)
        output_state = _deserialize_state(blamed.output_state)
        state_diff = compute_state_diff(input_state, output_state)

        prompt = self._build_prompt(blamed, state_diff, blame_result.reason)

        try:
            raw_response = self._call_llm(prompt)
            return DiagnosisResult(
                session_id=self._session_id,
                blamed_node_name=blamed.node_name,
                root_cause=raw_response.get("root_cause", "Diagnosis unavailable"),
                fix_suggestions=raw_response.get("fix_suggestions", []),
                confidence=raw_response.get("confidence", "low"),
                raw_response=json.dumps(raw_response),
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.warning(f"Diagnosis failed: {e}")
            return DiagnosisResult(
                session_id=self._session_id,
                blamed_node_name=blamed.node_name,
                root_cause="Diagnosis unavailable",
                fix_suggestions=[],
                confidence="low",
            )

    def _build_prompt(
        self,
        blamed_node: NodeExecution,
        state_diff: StateDiff,
        blame_reason: str,
    ) -> str:
        """Build the diagnosis prompt for the LLM.

        Args:
            blamed_node: The blamed NodeExecution.
            state_diff: State diff for the blamed node.
            blame_reason: The blame reason string.

        Returns:
            Formatted prompt string.
        """
        input_state = _deserialize_state(blamed_node.input_state)
        output_state = _deserialize_state(blamed_node.output_state)

        input_json = json.dumps(input_state, indent=2, default=str)[:500]
        output_json = json.dumps(output_state, indent=2, default=str)[:500]

        diff_lines = []
        for key, val in state_diff.added.items():
            diff_lines.append(f"  + {key}: {val}")
        for key, val in state_diff.removed.items():
            diff_lines.append(f"  - {key}: {val}")
        for key, vals in state_diff.modified.items():
            diff_lines.append(f"  ~ {key}: {vals.get('before')} -> {vals.get('after')}")
        diff_str = "\n".join(diff_lines) if diff_lines else "  (no changes)"

        return f"""\
Analyze this LangGraph node execution and explain why it caused a failure.

## Node Information
- Name: {blamed_node.node_name}
- Status: {blamed_node.status}
- Error: {blamed_node.error_message or "none"}
- Duration: {blamed_node.duration_ms}ms

## Blame Reason
{blame_reason}

## Input State
```json
{input_json}
```

## Output State
```json
{output_json}
```

## State Diff
{diff_str}

Respond in JSON:
{{
    "root_cause": "one paragraph explanation",
    "fix_suggestions": ["fix 1", "fix 2", "fix 3"],
    "confidence": "high|medium|low"
}}"""

    def _call_llm(self, prompt: str) -> dict:
        """Call the configured LLM provider.

        Supports groq, openai, and anthropic. Retries up to 2 times.

        Args:
            prompt: The user prompt to send.

        Returns:
            Parsed JSON dict from the LLM response.

        Raises:
            RuntimeError: If all retries fail.
        """
        last_error = None
        for attempt in range(3):
            try:
                if self._provider == "groq":
                    return self._call_groq(prompt)
                elif self._provider == "openai":
                    return self._call_openai(prompt)
                elif self._provider == "anthropic":
                    return self._call_anthropic(prompt)
                else:
                    raise ValueError(f"Unsupported provider: {self._provider}")
            except Exception as e:
                last_error = e
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                import traceback
                traceback.print_exc()
                if attempt < 2:
                    time.sleep(1 * (attempt + 1))

        return {
            "root_cause": f"Diagnosis failed: {last_error}",
            "fix_suggestions": [],
            "confidence": "low",
        }

    def _call_groq(self, prompt: str) -> dict:
        """Call Groq API."""
        import httpx

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")

        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return json.loads(content)

    def _call_openai(self, prompt: str) -> dict:
        """Call OpenAI API."""
        import httpx

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return json.loads(content)

    def _call_anthropic(self, prompt: str) -> dict:
        """Call Anthropic API."""
        import httpx

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": self._model,
                "max_tokens": 1024,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["content"][0]["text"]
        return json.loads(content)
