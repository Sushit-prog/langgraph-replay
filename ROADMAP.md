# AgentTrace Roadmap

## Shipped

- **Phase 1 — Span Annotation Layer**: SQLite-backed human judgments on execution steps, CLI for add/list/export.
- **Phase 2 — Regression Watchdog (CI Gate)**: Compare new runs against pinned baselines, fail CI on annotated-step regressions.
- **Phase 3 — State Similarity Loop Classifier**: Detect stuck loops vs. legitimate retries using sentence embeddings and progress heuristics.
- **Phase 4 — Semantic State Diffing**: Opt-in `--semantic` mode to reduce false-positive regressions from wording changes.
- **Phase 5 — Upstream Divergence Detection**: Trace regression causes backward through ancestor steps to find changed tool outputs and context.
- **Phase 6 — Counterfactual Replay**: Test causal hypotheses by replaying graphs with substituted baseline values at forked checkpoints.
- **Phase 7 — Wiring and Polish**: `--from-divergence` flag for counterfactual CLI, demo fixture, this roadmap.

## Deliberately Deferred

- **Flakiness scorer**: Classifies runs by repeatability (runs that sometimes pass, sometimes fail on identical inputs). Deferred because it requires running the same input multiple times, which is expensive on CPU-only hardware and adds CI wall-clock time that most teams aren't willing to pay for yet.
- **Multi-backend OTel export**: Export traces in OpenTelemetry format for ingestion into Grafana/Datadog/etc. Deferred because the current local SQLite store serves the primary use case (CLI-first debugging), and OTel integration would require maintaining compatibility with a shifting vendor spec — checkbox-feature risk outweighs benefit at this stage.
- **Trace schema validator**: Validates that recorded traces conform to a declared schema before analysis. Deferred because it's a side-quest — the existing trace recording is deterministic and the schema is implicit in the `NodeExecution` model, so a formal validator would be over-engineering for the current scale.

## Explicitly Out of Scope

- **Web DAG view / hosted UI**: CLI/CI-native is the product; a hosted UI would be a different pitch requiring backend infrastructure, auth, and a deployment model that doesn't match how this tool is used (local debugging, CI pipelines).
