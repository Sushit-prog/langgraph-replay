"""Blame analysis for identifying which node caused a failure."""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel
from rich.console import Console

from langgraph_replay.storage import NodeExecution, ReplayStorage, _deserialize_state

# Lazy import for diagnosis — resolved at call time to avoid circular import
DiagnosisEngine = None
DiagnosisResult = None

logger = logging.getLogger(__name__)


@dataclass
class NodeAnalysis:
    """Analysis result for a single node."""

    node_name: str
    execution_order: int
    status: str
    issues_found: list[str] = field(default_factory=list)
    is_blamed: bool = False


class BlameResult(BaseModel):
    """Result of blame analysis."""

    blamed_node: Optional[NodeExecution] = None
    reason: str = ""
    confidence: str = "low"
    analysis: list[NodeAnalysis] = []
    diagnosis: Optional[Any] = None


class BlameEngine:
    """Identifies which node caused a failure by walking backwards.

    Structural blame logic:
    1. Look at the session's final output state.
    2. For each key in the initial input state that is missing from the final
       output, find which node was the last to have that key in its input_state
       but not its output_state. That is the blamed node for that key.
    3. Additionally flag any node where status == "error".
    4. If no keys are dropped and no errors exist, blamed_node is None.

    Confidence rules:
    - "high" if status == "error"
    - "high" if a key disappears and never reappears after that node
    - "medium" if a key disappears but the final output is non-empty
    - "low" if blame is inferred from incomplete state only

    Usage::

        engine = BlameEngine("session_abc123")
        result = engine.run()
        if result.blamed_node:
            print(f"Blame: {result.blamed_node.node_name} - {result.reason}")
    """

    def __init__(self, session_id: str, storage: Optional[ReplayStorage] = None):
        """Initialize the blame engine.

        Args:
            session_id: The session ID to analyze.
            storage: Optional ReplayStorage instance.
        """
        self._storage = storage or ReplayStorage()
        self._session_id = session_id
        self._session = self._storage.get_session(session_id)
        self._executions = self._storage.get_node_executions(session_id)

    def run(
        self,
        baseline_session_id: Optional[str] = None,
        use_eval: bool = False,
        diagnose: bool = False,
    ) -> BlameResult:
        """Run blame analysis on the session.

        Args:
            baseline_session_id: If provided, enables semantic comparison.
            use_eval: If True and pytest-llm installed, uses semantic blame.
            diagnose: If True and blamed_node found, run LLM diagnosis.

        Returns:
            BlameResult with blamed node, reason, confidence, and analysis.
        """
        if not self._executions:
            return BlameResult(reason="No node executions found")

        pytest_llm = None
        if use_eval:
            pytest_llm = self._try_import_pytest_llm()

        # Auto-baseline selection when use_eval is True but no baseline provided
        _console = Console()
        if baseline_session_id is None and use_eval:
            candidates = self._storage.search_sessions(
                agent_name=self._session.agent_name,
                status="completed",
            )
            candidates = [s for s in candidates if s.id != self._session_id]
            if candidates:
                baseline_session_id = candidates[0].id
                _console.print(
                    f"[yellow]No baseline provided. Using most recent "
                    f"completed session as baseline: {baseline_session_id}[/yellow]"
                )
            else:
                _console.print(
                    f"[yellow]No baseline session found for agent "
                    f"'{self._session.agent_name}'. Running structural blame only.[/yellow]"
                )

        baseline_execs = []
        if baseline_session_id:
            baseline_execs = self._storage.get_node_executions(baseline_session_id)

        # Get initial input keys (from first node's input)
        initial_input = _deserialize_state(self._executions[0].input_state)
        initial_keys = {k for k in initial_input.keys() if not k.startswith("_")}

        # Get final output keys (from last node's output)
        last_exec = self._executions[-1]
        final_output = _deserialize_state(last_exec.output_state)
        final_keys = set(final_output.keys())

        # Collect all keys that ever appeared in any node's output
        ever_in_output = set()
        for execution in self._executions:
            out = _deserialize_state(execution.output_state)
            ever_in_output.update(out.keys())

        # Keys missing from final output — only if they ever appeared
        # in some node's output (input-only keys are not "dropped")
        missing_keys = (initial_keys - final_keys) & ever_in_output

        # Build analysis for each node
        analysis_results = []
        blamed = None
        blame_reasons = []
        semantic_blame_applied = False

        for execution in self._executions:
            issues = []

            # Check for error status
            if execution.status == "error":
                issues.append(f"Node raised error: {execution.error_message}")

            output_state = _deserialize_state(execution.output_state)
            input_state = _deserialize_state(execution.input_state)

            # For each missing key, check if this node dropped it
            # A node "drops" a key if the key was in its input but not its output
            # AND no later node has it in its output (it never reappears)
            for key in missing_keys:
                if key in input_state and key not in output_state:
                    # Check if this key ever reappears in later nodes
                    key_reappears = False
                    for later_exec in self._executions[execution.execution_order + 1:]:
                        later_output = _deserialize_state(later_exec.output_state)
                        if key in later_output:
                            key_reappears = True
                            break

                    if not key_reappears:
                        issues.append(
                            f"Key '{key}' dropped and never recovered"
                        )

            # MODE 2: Semantic blame (if pytest-llm available)
            if pytest_llm and baseline_execs:
                baseline_by_name = {e.node_name: e for e in baseline_execs}
                if execution.node_name in baseline_by_name:
                    baseline_exec = baseline_by_name[execution.node_name]
                    baseline_output = _deserialize_state(baseline_exec.output_state)
                    for key, val in output_state.items():
                        if key.startswith("_"):
                            continue
                        if isinstance(val, str) and len(val) > 20:
                            baseline_val = baseline_output.get(key, "")
                            if isinstance(baseline_val, str) and len(baseline_val) > 20:
                                try:
                                    pytest_llm.assert_regression(
                                        output=val,
                                        baseline=baseline_val,
                                        threshold=0.75,
                                    )
                                except AssertionError as e:
                                    blamed = execution
                                    confidence = "high"
                                    reason = str(e)
                                    semantic_blame_applied = True
                                    issues.append(f"Semantic regression: {e}")

            analysis = NodeAnalysis(
                node_name=execution.node_name,
                execution_order=execution.execution_order,
                status=execution.status,
                issues_found=issues,
            )
            analysis_results.append(analysis)

            # Collect reasons for blaming this node
            for issue in issues:
                blame_reasons.append(f"Node '{execution.node_name}': {issue}")

        # Determine the blamed node
        # Priority 1: First node with status == "error" (walking forward)
        for execution in self._executions:
            if execution.status == "error":
                blamed = execution
                for a in analysis_results:
                    if a.node_name == execution.node_name:
                        a.is_blamed = True
                        break
                break

        # Priority 2: If no error and no semantic blame, find the first node that dropped a key
        if blamed is None and missing_keys and not semantic_blame_applied:
            for execution in self._executions:
                output_state = _deserialize_state(execution.output_state)
                input_state = _deserialize_state(execution.input_state)
                for key in missing_keys:
                    if key in input_state and key not in output_state:
                        # Check if key ever reappears
                        key_reappears = False
                        for later_exec in self._executions[execution.execution_order + 1:]:
                            later_output = _deserialize_state(later_exec.output_state)
                            if key in later_output:
                                key_reappears = True
                                break
                        if not key_reappears:
                            blamed = execution
                            for a in analysis_results:
                                if a.node_name == execution.node_name:
                                    a.is_blamed = True
                                    break
                            break
                if blamed is not None:
                    break

        # Determine confidence and reason
        if not semantic_blame_applied:
            confidence = "low"
            reason = "No issues found"

        if blamed and not semantic_blame_applied:
            if blamed.status == "error":
                confidence = "high"
                reason = f"Node '{blamed.node_name}' raised an error: {blamed.error_message}"
            elif missing_keys:
                # Check if any dropped key never reappeared
                dropped_permanently = False
                for key in missing_keys:
                    output_state = _deserialize_state(blamed.output_state)
                    input_state = _deserialize_state(blamed.input_state)
                    if key in input_state and key not in output_state:
                        key_reappears = False
                        for later_exec in self._executions[blamed.execution_order + 1:]:
                            later_output = _deserialize_state(later_exec.output_state)
                            if key in later_output:
                                key_reappears = True
                                break
                        if not key_reappears:
                            dropped_permanently = True
                            break

                if dropped_permanently:
                    confidence = "high"
                elif final_output:
                    confidence = "medium"
                else:
                    confidence = "low"

                reason_parts = []
                for a in analysis_results:
                    if a.is_blamed and a.issues_found:
                        reason_parts.append(
                            f"Node '{blamed.node_name}' has issues: {'; '.join(a.issues_found)}"
                        )
                reason = reason_parts[0] if reason_parts else f"Node '{blamed.node_name}' is suspect"

        result = BlameResult(
            blamed_node=blamed,
            reason=reason,
            confidence=confidence,
            analysis=analysis_results,
        )

        # Run diagnosis if requested and a blamed node was found
        if diagnose and blamed is not None:
            try:
                from langgraph_replay.diagnosis import DiagnosisEngine as _DiagnosisEngine
                diag_engine = _DiagnosisEngine(
                    session_id=self._session_id,
                    storage=self._storage,
                )
                result.diagnosis = diag_engine.diagnose(result)
            except Exception as e:
                logger.warning(f"Diagnosis failed: {e}")

        return result

    def _try_import_pytest_llm(self) -> Optional[Any]:
        """Attempt to import pytest_llm.assertions.

        Returns:
            The module if available, None otherwise.
        """
        try:
            import pytest_llm.assertions
            return pytest_llm.assertions
        except ImportError:
            return None
