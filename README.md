# tiny-eval

> **Zero-dependency agent evaluation framework.** Define benchmark suites, grade agent runs, track regressions over time. MIT.

[![Install from GitHub](https://img.shields.io/badge/install-git%2Bhttps-blue)](https://github.com/hussain-alsaibai/tiny-eval#install)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://github.com/hussain-alsaibai/tiny-eval#install)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Zero deps](https://img.shields.io/badge/dependencies-0-brightgreen)](https://github.com/hussain-alsaibai/tiny-eval#install)

## Why?

AI agents need eval suites the same way software needs test suites. "Does the agent work?" stopped being a vibe check in 2025. In 2026, it's a CI gate.

`tiny-eval` gives you:
- **Benchmark suites** — define cases with prompts, scoring functions, thresholds
- **Agent grading** — run any agent function against a suite, get per-case scores
- **Regression tracking** — compare each run against history, detect score drops
- **CI integration** — block merges on eval regression, just like test suites
- **Zero dependencies** — pure stdlib, audit in an afternoon

## Install

```bash
pip install tiny-eval
```

Or grab the single file:
```bash
curl -O https://raw.githubusercontent.com/hussain-alsaibai/tiny-eval/main/tiny_eval.py
```

## Quickstart

```python
from tiny_eval import EvalSuite, EvalCase, evaluate

# ── Option 1: Full suite ──────────────────────────────────────
suite = EvalSuite(
    name="bug-fix",
    version="1.0.0",
    threshold=0.7,
    description="Agent must fix bugs without crashing",
    cases=[
        EvalCase(
            "no_crash",
            "Must not crash on valid input",
            input_fn=lambda: "Write a hello world in Python",
            expected_fn=lambda out: 1.0 if out and not out.get("crashed") else 0.0,
            tags=["safety"],
        ),
        EvalCase(
            "has_output",
            "Must return non-empty result",
            input_fn=lambda: "Return the string 'hello'",
            expected_fn=lambda out: 1.0 if out and str(out).strip() else 0.0,
            tags=["correctness"],
        ),
    ],
)

# Your agent: fn(prompt) -> output
def my_agent(prompt: str):
    return {"response": "hello", "crashed": False}

run = suite.run(my_agent, agent_info={"model": "gpt-4o", "version": "2.0"})
print(run.summary())
# ✅ PASS | bug-fix@1.0.0 | 2/2 cases | 100% | score 2.0/2.0 | 45ms

# ── Option 2: Ad-hoc evaluate() ──────────────────────────────
result = evaluate(
    my_agent,
    [
        ("no_crash", "Must not crash", lambda o: 1.0 if o else 0.0),
        ("has_content", "Must return content", lambda o: 1.0 if str(o).strip() else 0.0),
    ],
)
print(result.summary())
```

## Features

| Feature | Description |
|---------|-------------|
| **Named suites** | Versioned, describable eval suites with metadata |
| **Flexible scoring** | Any `output -> float` function; 0–1 normalized |
| **Regression tracking** | Record runs to JSON, compare against history |
| **Threshold gating** | Per-suite threshold; scores below block the run |
| **Agent info** | Attach model name, version, config to every run |
| **CLI helpers** | `print_run_summary()` for terminal output |
| **Tags** | Tag cases (safety, correctness, speed) for filtering |
| **Zero deps** | Pure Python stdlib; works on Python 3.9+ |

## Regression Tracking

```python
from tiny_eval import EvalSuite, EvalCase, RegressionTracker

tracker = RegressionTracker("runs.json")

# Run your agent
run = suite.run(my_agent)
tracker.record(run)

# Check for regression
best = tracker.best("bug-fix")
print(f"Best score: {best.total_score:.1f}")

# History
for r in tracker.history("bug-fix"):
    print(f"  {r.timestamp} -> {r.total_score:.1f} ({'PASS' if r.passed else 'FAIL'})")
```

### CI Gate Pattern

```python
# In your CI pipeline (GitHub Actions, etc.)
run = suite.run(agent_fn, agent_info={"model": "gpt-4o"})
tracker = RegressionTracker("eval_runs.json")
result = tracker.record(run)

if result.regression:
    print(f"::error::Regression detected: {result.delta:.2f} vs previous")
    exit(1)  # Block the PR

if not run.passed():
    print(f"::error::Eval suite failed: {run.pass_rate:.0%} below threshold")
    exit(1)
```

## Architecture

```
EvalSuite
  └─ list[EvalCase]
        input_fn()  →  agent_fn(prompt)  →  expected_fn(output)  →  score ∈ [0,1]

EvalRun
  └─ list[EvalResult]
        case_id, passed, score, threshold, duration_ms, error

RegressionTracker
  └─ list[RegressionRecord]
        Compare run against best/previous → regression delta
```

## Comparison

| Feature | tiny-eval | custom scripts | langsmith evals |
|---------|-----------|----------------|-----------------|
| **Dependencies** | 0 | varies | 5+ |
| **Self-hosted** | ✅ | ✅ | ❌ |
| **Regression tracking** | ✅ | ❌ | ⚠️ |
| **CI gate built-in** | ✅ | ❌ | ⚠️ |
| **Single file** | ✅ | ⚠️ | ❌ |
| **Suite versioning** | ✅ | ❌ | ✅ |
| **MIT License** | ✅ | varies | proprietary |

## License

MIT © 2026 Hussain Al-Saibai. See [LICENSE](LICENSE).
