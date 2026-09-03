# tiny-eval

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Dependencies](https://img.shields.io/badge/Dependencies-zero-brightgreen)

> **Zero-dependency agent evaluation framework.** Benchmark suites, grading rubrics, regression tracking. For autonomous AI workflows.

`tiny-eval` is a single-file Python library (~400 lines, no external imports) for running structured evaluations of AI agents. Define benchmark tasks with custom grading rubrics, track scores over time, detect regressions, and integrate everything into CI — in under 5 minutes.

---

## Features

- **Benchmark suite definition** — Structured `BenchmarkSuite` with `task/expected/criteria` for every case
- **Runnable benchmark cases** — Pass/fail grading per case with detailed criteria scores
- **Grading rubric with weighted criteria** — Combine multiple named criteria with custom weights
- **Regression tracking over time** — JSONL persistence, baseline management, historical comparison
- **Pre-built suite templates** — `bug-fix`, `code-quality`, `security`, `context-efficiency` included
- **Agent runner wrapper** — Execute an agent, capture timing, cost, and errors
- **Statistics** — Pass rate, average score, total duration, cost per run
- **Diff between two runs** — `RegressionReport` surfaces new failures, improvements, and regressions
- **CLI** — `tiny-eval run <suite>` and `tiny-eval compare a.json b.json`
- **Zero external dependencies** — Standard library only, MIT licensed
- **CI integration** — Block pipelines on regression with `check_regression()`

---

## Quick Start

```python
from tiny_eval import (
    BenchmarkSuite, BenchmarkCase, Rubric, Criterion,
    Score, AgentRunner,
)

# 1. Define how to grade a result
def is_correct(task, result):
    return Score(
        value=1.0 if result == task["expected"] else 0.0,
        reason="Output matches expected" if result == task["expected"] else "Mismatch",
    )

rubric = Rubric([
    Criterion("correctness", "Output must match expected", weight=1.0, evaluate_fn=is_correct),
])

# 2. Define a task
def task_fn():
    return {"question": "2 + 2", "expected": 4}

# 3. Build a suite
case = BenchmarkCase(
    task_id="add",
    description="Basic addition",
    task_fn=task_fn,
    expected_fn=lambda: 4,
    rubric=rubric,
)
suite = BenchmarkSuite("math", [case])

# 4. Run against any agent
def my_agent(task):
    # Your AI call here
    return 4

result = suite.run(my_agent)
print(result.pass_rate, result.avg_score)  # 1.0 1.0
```

---

## Defining a Benchmark Suite

```python
from tiny_eval import BenchmarkSuite, BenchmarkCase, Rubric, Criterion, Score

def my_agent(task):
    # Call your LLM here
    return "generated output"

rubric = Rubric([
    Criterion("accuracy", "Output is factually correct", 1.0,
              lambda t, r: Score(1.0 if r == t["expected"] else 0.0, "match")),
    Criterion("length", "Output is concise (<50 chars)", 0.3,
              lambda t, r: Score(1.0 if len(str(r)) < 50 else 0.5, "length ok")),
])

suite = BenchmarkSuite("my-eval", description="My evaluation suite")
suite.add_case(BenchmarkCase(
    task_id="task-1",
    description="Evaluate fact correctness",
    task_fn=lambda: {"expected": "Paris"},
    expected_fn=lambda: "Paris",
    rubric=rubric,
))
```

---

## Running Eval Suites

### Via Python API

```python
from tiny_eval import AgentRunner, bug_fix_suite

runner = AgentRunner(
    agent_fn=lambda task: "def fix(): pass",  # your agent
    cost_fn=lambda task: 0.02,               # optional cost hook
)

result = runner.run_suite(bug_fix_suite(), verbose=True)
print(f"Pass rate: {result.pass_rate:.1%}")
```

### Via CLI

```bash
# Run a pre-built suite
python tiny_eval.py run bug-fix --verbose
python tiny_eval.py run code-quality --verbose
python tiny_eval.py run security --verbose
python tiny_eval.py run context-efficiency --verbose

# Run a custom suite from a Python file
python tiny_eval.py run my_suite.py --verbose
```

---

## Regression Tracking

```python
import json
from tiny_eval import RegressionTracker, AgentRunner, bug_fix_suite

tracker = RegressionTracker("runs.jsonl")   # persisted to JSONL
runner = AgentRunner(my_agent)

# First run — record as baseline
result = runner.run_suite(bug_fix_suite())
run_id = tracker.record("bug-fix", result, baseline=True)
print(f"Baseline recorded: {run_id}")

# Next day — new code, new run
result2 = runner.run_suite(bug_fix_suite())
run_id2 = tracker.record("bug-fix", result2)

# Compare
report = tracker.compare("bug-fix", run_id, run_id2)
print(f"Delta: {report.delta_score:+.3f}")

# Check for regression in CI
if tracker.check_regression("bug-fix", result2):
    print("REGRESSION — blocking pipeline!")
    exit(1)
```

---

## CI Integration (GitHub Actions)

```yaml
name: Agent Eval

on:
  push:
    branches: [main]
  pull_request:

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install tiny-eval
        run: pip install .

      - name: Run evaluation
        id: eval
        run: |
          python my_eval_runner.py --output results.json

      - name: Check for regression
        run: |
          python -c "
          import json, sys
          from tiny_eval import RegressionTracker, SuiteResult

          tracker = RegressionTracker('runs.jsonl')
          with open('results.json') as f:
              result = SuiteResult.from_dict(json.load(f))

          if tracker.check_regression('bug-fix', result):
              print('REGRESSION DETECTED — failing build')
              sys.exit(1)
          print('No regression — all good')
          "

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: results.json
```

---

## Comparison Table

| Feature                      | tiny-eval | LangSmith Eval | Helicone Eval | Custom Script |
|------------------------------|-----------|----------------|---------------|---------------|
| Zero dependencies            | ✅        | ❌             | ❌            | ✅            |
| Single-file library          | ✅        | ❌             | ❌            | ✅            |
| Custom rubric grading        | ✅        | ✅             | ❌            | ✅            |
| Regression tracking          | ✅        | ✅             | ✅            | ❌            |
| Built-in benchmark suites    | ✅        | ❌             | ❌            | ❌            |
| CI/regression blocking       | ✅        | ✅             | ✅            | ✅            |
| Self-hosted / no vendor lock | ✅        | ❌             | ❌            | ✅            |
| CLI included                 | ✅        | ✅             | ❌            | ❌            |
| JSONL persistence            | ✅        | ❌             | ❌            | ✅            |
| Python-only                  | ✅        | ❌             | ❌            | ✅            |

---

## Ecosystem

| Tool             | Description                                         |
|------------------|-----------------------------------------------------|
| **tiny-eval**    | This library — zero-dep agent evaluation            |
| **LangSmith**    | SaaS eval platform by LangChain                     |
| **Helicone**     | Observability & eval for AI apps                    |
| **RAGAS**        | Specialized RAG pipeline evaluation                |
| **OpenAI Evals** | OpenAI's open-source evaluation framework           |
| **Braintrust**   | AI evals and regression testing platform            |
| **Phoenix**      | Arize open-source LLM observability                 |

---

## API Reference

### Data Classes

| Class             | Fields                                                                 |
|-------------------|------------------------------------------------------------------------|
| `Score`           | `value: float`, `reason: str`, `metadata: dict`                        |
| `EvalResult`      | `task_id`, `passed`, `score`, `criteria_scores`, `duration_ms`, `cost`, `error`, `timestamp` |
| `SuiteResult`     | `suite_name`, `results`, `total_duration_ms`, `total_cost`, `pass_rate`, `avg_score`, `started_at`, `completed_at` |
| `RegressionReport`| `suite_name`, `baseline_run`, `current_run`, `baseline_score`, `current_score`, `delta_score`, `new_failures`, `improved`, `regressions` |
| `Criterion`       | `name`, `description`, `weight`, `evaluate_fn`                         |
| `BenchmarkCase`   | `task_id`, `description`, `task_fn`, `expected_fn`, `rubric`, `metadata` |

### Classes

| Class             | Key Methods                                                      |
|-------------------|------------------------------------------------------------------|
| `Rubric`          | `evaluate(task, result)`, `add_criterion(criterion)`, `summary()` |
| `BenchmarkSuite`  | `run(agent_fn, verbose)`, `add_case(case)`, `get_case(id)`, `total_cases` |
| `RegressionTracker`| `record()`, `get_runs()`, `compare()`, `get_baseline()`, `set_baseline()`, `check_regression()` |
| `AgentRunner`     | `run(task)`, `run_suite(suite, verbose)`                        |
| `TinyEvalCLI`     | `run_suite(name_or_path)`, `compare(a, b)`, `report(result)`    |

### Pre-built Suites

| Function                   | Suite Name           | Criteria                                          |
|----------------------------|----------------------|---------------------------------------------------|
| `bug_fix_suite()`          | `bug-fix`            | compiles, no_secrets, no_crash                   |
| `code_quality_suite()`     | `code-quality`       | syntax, lint, error_handling, logging, types     |
| `security_suite()`         | `security`           | no_secrets, sql_injection, cmd_injection, validation |
| `context_efficiency_suite()`| `context-efficiency` | context_window, tool_calls, token_budget          |

### Rubric Threshold

A case **passes** if its weighted score is `>= 0.7`. You can change this by subclassing `Rubric` and overriding `evaluate()`.

---

## Installation

```bash
pip install .
# or in CI:
pip install git+https://github.com/hussain-alsaibai/tiny-eval.git
```

## License

MIT © Hussain Al-Saibai
