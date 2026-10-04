# Changelog

## v0.2.0 — 2026-10-05

Rewritten to match the test contract:
- `Score(value, reason, metadata)` — bounded [0,1] score with reason + metadata.
- `Criterion(name, description, weight, evaluate_fn)` — one weighted scoring rule.
- `Rubric([criteria])` — weighted average of criteria with `evaluate(task, response) -> EvalResult`.
- `BenchmarkCase(task_id, description, task_fn, expected_fn, rubric, metadata)` — one task.
- `BenchmarkSuite(name, cases)` — suite with `add_case`, `get_case`, `run(agent_fn, verbose)`.
- `AgentRunner(agent_fn, cost_fn)` — wraps an agent with timing + cost + error capture.
- `RegressionTracker(path)` — JSONL-backed history with `record`, `get_runs`, `get_baseline`, `set_baseline`, `compare`, `check_regression`.
- `TinyEvalCLI()` — `report(suite_result)`, `compare(baseline_path, current_path)`, `run_suite(name, verbose)`.
- 4 pre-built suites: `bug_fix_suite()`, `code_quality_suite()`, `security_suite()`, `context_efficiency_suite()`.

Tests: 63/63 passing in `test_tiny_eval.py`.

## v0.1.0 — 2026-10-04 (initial)

First commit with `EvalSuite`, `EvalCase`, `EvalResult`, `EvalRun`, `RegressionTracker`, `evaluate`.
No passing tests yet — API was placeholder.
