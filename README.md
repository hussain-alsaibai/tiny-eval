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
from tiny_eval_v100 import EvalSuite, Criterion

suite = EvalSuite(
    "bug-fix",
    [
        Criterion("no_crash",   "Must not crash on valid input",
                  lambda ctx: not ctx.get("crashed")),
        Criterion("has_output", "Must return a non-empty result",
                  lambda ctx: bool(str(ctx.get("response", "")).strip())),
    ],
    pass_threshold=0.7,
)

# `context` is whatever your agent produced for this run
result = suite.evaluate(
    agent_name="my-agent",
    run_id="run-001",
    context={"response": "hello", "crashed": False},
)

print(result.grade)          # Grade.PASS
print(result.overall_score)  # 1.0
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
from tiny_eval import RegressionTracker

tracker = RegressionTracker("runs.json")

# `result` is a SuiteResult produced by a suite run
tracker.record("bug-fix", result, baseline=True)

# Later: has the score dropped more than 5%?
regressed = tracker.check_regression("bug-fix", result, drop_threshold=0.05)
print("regressed:", regressed)

best = tracker.get_baseline("bug-fix")
for run in tracker.get_runs("bug-fix", limit=5):
    print(" ", run)
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
