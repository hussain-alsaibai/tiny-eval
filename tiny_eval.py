"""
tiny_eval.py — Zero-Dependency Agent Evaluation Framework

A ~400-line library for evaluating AI agent behavior with benchmark suites,
grading rubrics, and regression tracking. No external dependencies.
"""

from __future__ import annotations

import dataclasses
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Score:
    """A single criterion score.

    Attributes:
        value:  Score value between 0.0 (fail) and 1.0 (perfect).
        reason: Human-readable explanation for the score.
        metadata: Optional arbitrary metadata (e.g., raw llm grading output).
    """

    value: float
    reason: str
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Score value must be in [0.0, 1.0], got {self.value}")


@dataclasses.dataclass
class EvalResult:
    """Result of evaluating a single benchmark case.

    Attributes:
        task_id:        Unique identifier for the task.
        passed:         True if the case passed (score >= 0.7 by default).
        score:          Overall weighted score in [0.0, 1.0].
        criteria_scores: List of individual criterion scores.
        duration_ms:    How long the agent took to produce the result.
        cost:           Estimated monetary cost of the agent call(s).
        error:          Error message if the agent failed, else None.
        timestamp:      ISO-8601 timestamp of when evaluation occurred.
    """

    task_id: str
    passed: bool
    score: float
    criteria_scores: list[Score]
    duration_ms: float
    cost: float = 0.0
    error: str | None = None
    timestamp: str = dataclasses.field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class SuiteResult:
    """Aggregated result of running an entire benchmark suite.

    Attributes:
        suite_name:        Name of the benchmark suite.
        results:           Per-case EvalResult list.
        total_duration_ms: Sum of all case durations.
        total_cost:         Sum of all case costs.
        pass_rate:          Fraction of cases that passed.
        avg_score:          Mean weighted score across cases.
        started_at:         ISO-8601 start timestamp.
        completed_at:       ISO-8601 completion timestamp.
    """

    suite_name: str
    results: list[EvalResult]
    total_duration_ms: float
    total_cost: float
    pass_rate: float
    avg_score: float
    started_at: str
    completed_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **dataclasses.asdict(self),
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SuiteResult:
        data = dict(data)
        data["results"] = [EvalResult(**r) for r in data["results"]]
        return cls(**data)


@dataclasses.dataclass
class RegressionReport:
    """Diff report between two eval runs.

    Attributes:
        suite_name:    Suite being compared.
        baseline_run:  Run ID used as baseline.
        current_run:   Run ID being evaluated.
        baseline_score: Baseline average score.
        current_score:  Current average score.
        delta_score:   current_score - baseline_score (negative = regression).
        new_failures:  Task IDs that passed on baseline but fail now.
        improved:      Task IDs that improved.
        regressions:   Task IDs that regressed.
    """

    suite_name: str
    baseline_run: str
    current_run: str
    baseline_score: float
    current_score: float
    delta_score: float
    new_failures: list[str]
    improved: list[str]
    regressions: list[str]


# ---------------------------------------------------------------------------
# Criterion & Rubric
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Criterion:
    """A single grading dimension within a rubric.

    Attributes:
        name:         Short identifier, e.g. "no_secrets".
        description:  Human-readable description, e.g. "No API keys in output".
        weight:       Importance weight (higher = more influence on final score).
                     Weights within a rubric do not need to sum to 1.
        evaluate_fn:  Callable(task, result) -> Score.
    """

    name: str
    description: str
    weight: float
    evaluate_fn: Callable[[Any, Any], Score]

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(f"Criterion weight must be non-negative, got {self.weight}")


class Rubric:
    """A collection of weighted criteria used to score a benchmark case.

    Example:
        rubric = Rubric([
            Criterion("correctness", "Output matches expected", 1.0,
                      lambda task, result: Score(1.0 if result == task["expected"] else 0.0,
                                                 "Match" if result == task["expected"] else "Mismatch")),
        ])
        eval_result = rubric.evaluate(task, result)
    """

    def __init__(self, criteria: list[Criterion]) -> None:
        if not criteria:
            raise ValueError("Rubric must have at least one criterion")
        self._criteria: list[Criterion] = list(criteria)

    def add_criterion(self, criterion: Criterion) -> None:
        """Add a criterion to the rubric after construction."""
        self._criteria.append(criterion)

    def evaluate(self, task: Any, result: Any) -> EvalResult:
        """Run all criteria against task and result.

        Args:
            task:   The task input (passed to each criterion's evaluate_fn).
            result: The agent's output (passed to each criterion's evaluate_fn).

        Returns:
            EvalResult with per-criterion scores and a weighted aggregate.
        """
        criteria_scores: list[Score] = []
        weighted_sum = 0.0
        total_weight = 0.0

        for criterion in self._criteria:
            try:
                score = criterion.evaluate_fn(task, result)
            except Exception as exc:  # pragma: no cover — criterion should not raise
                score = Score(0.0, f"Criterion '{criterion.name}' raised: {exc}")

            criteria_scores.append(score)
            weighted_sum += score.value * criterion.weight
            total_weight += criterion.weight

        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        passed = final_score >= 0.7

        return EvalResult(
            task_id=getattr(task, "task_id", "unknown"),
            passed=passed,
            score=final_score,
            criteria_scores=criteria_scores,
            duration_ms=0.0,
        )

    def summary(self) -> list[tuple[str, str, float]]:
        """Return (name, description, weight) for every criterion."""
        return [(c.name, c.description, c.weight) for c in self._criteria]

    @property
    def criteria(self) -> list[Criterion]:
        """Read-only view of criteria."""
        return list(self._criteria)


# ---------------------------------------------------------------------------
# Benchmark Suite
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BenchmarkCase:
    """A single evaluable task within a suite.

    Attributes:
        task_id:     Unique string ID within the suite.
        description: Human-readable task description.
        task_fn:     Callable returning the task input.
        expected_fn: Callable returning the expected output.
        rubric:      Rubric instance defining how to grade.
        metadata:    Arbitrary metadata (e.g., difficulty, tags).
    """

    task_id: str
    description: str
    task_fn: Callable[[], Any]
    expected_fn: Callable[[], Any]
    rubric: Rubric
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


class BenchmarkSuite:
    """A named collection of benchmark cases evaluated against an agent.

    Example:
        suite = BenchmarkSuite("my-suite", [case1, case2])
        result = suite.run(my_agent_fn)
        print(result.pass_rate, result.avg_score)
    """

    def __init__(
        self,
        name: str,
        cases: list[BenchmarkCase] | None = None,
        description: str = "",
    ) -> None:
        self.name = name
        self.description = description
        self._cases: dict[str, BenchmarkCase] = {}
        if cases:
            for case in cases:
                self.add_case(case)

    def add_case(self, case: BenchmarkCase) -> None:
        """Register a benchmark case by its task_id."""
        if case.task_id in self._cases:
            raise ValueError(f"Case with task_id '{case.task_id}' already exists in suite")
        self._cases[case.task_id] = case

    def get_case(self, task_id: str) -> BenchmarkCase | None:
        """Retrieve a case by task_id, or None if not found."""
        return self._cases.get(task_id)

    @property
    def total_cases(self) -> int:
        """Number of cases in the suite."""
        return len(self._cases)

    def run(
        self,
        agent_fn: Callable[[Any], Any],
        verbose: bool = False,
    ) -> SuiteResult:
        """Evaluate the agent against every case in the suite.

        Args:
            agent_fn: Callable that accepts a task input and returns the agent's output.
            verbose:  If True, print progress to stdout.

        Returns:
            SuiteResult with per-case EvalResult objects and aggregate statistics.
        """
        results: list[EvalResult] = []
        started_at = datetime.now(timezone.utc).isoformat()

        for case in self._cases.values():
            if verbose:
                print(f"  [{case.task_id}] {case.description}...", end=" ", flush=True)

            task = case.task_fn()
            error: str | None = None
            duration_ms = 0.0
            cost = 0.0
            raw_result: Any = None

            try:
                start = time.perf_counter()
                raw_result = agent_fn(task)
                duration_ms = (time.perf_counter() - start) * 1000
            except Exception as exc:  # pragma: no cover — agent should not raise
                error = f"{type(exc).__name__}: {exc}"
                raw_result = None

            # Build task object that rubric expects (inject task_id for EvalResult)
            class _Task:
                pass

            _task = _Task()
            _task.task_id = case.task_id
            for k, v in (task or {}).items() if isinstance(task, dict) else {}:
                setattr(_task, k, v)

            if error:
                result_obj = EvalResult(
                    task_id=case.task_id,
                    passed=False,
                    score=0.0,
                    criteria_scores=[],
                    duration_ms=duration_ms,
                    error=error,
                )
            else:
                result_obj = case.rubric.evaluate(_task, raw_result)
                result_obj.duration_ms = duration_ms
                result_obj.cost = cost

            results.append(result_obj)

            if verbose:
                status = "PASS" if result_obj.passed else "FAIL"
                print(f"{status} ({result_obj.score:.2f}, {duration_ms:.0f}ms)")

        completed_at = datetime.now(timezone.utc).isoformat()
        total_duration = sum(r.duration_ms for r in results)
        total_cost = sum(r.cost for r in results)
        passed_count = sum(1 for r in results if r.passed)
        avg_score = sum(r.score for r in results) / len(results) if results else 0.0

        return SuiteResult(
            suite_name=self.name,
            results=results,
            total_duration_ms=total_duration,
            total_cost=total_cost,
            pass_rate=passed_count / len(results) if results else 0.0,
            avg_score=avg_score,
            started_at=started_at,
            completed_at=completed_at,
        )


# ---------------------------------------------------------------------------
# Regression Tracker
# ---------------------------------------------------------------------------


class RegressionTracker:
    """Persists SuiteResult runs to a JSON file and supports regression queries.

    Storage format (one JSON object per line, gzipped):
        {"run_id": "...", "suite_name": "...", "result": {...}, "baseline": false}

    Example:
        tracker = RegressionTracker("runs.jsonl")
        tracker.record("math-suite", suite_result)
        if tracker.check_regression("math-suite", new_result):
            print("REGRESSION DETECTED!")
    """

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self._path = Path(storage_path) if storage_path else Path.cwd() / "eval_runs.jsonl"
        self._cache: list[dict[str, Any]] = []
        self._baselines: dict[str, str] = {}  # suite_name -> run_id
        self._load()

    # -- Persistence ---------------------------------------------------------

    def _load(self) -> None:
        """Load runs from the JSONL file into memory."""
        if not self._path.exists():
            return
        self._cache = []
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        self._cache.append(entry)
                        if entry.get("baseline"):
                            self._baselines[entry["suite_name"]] = entry["run_id"]
        except (json.JSONDecodeError, IOError):  # pragma: no cover
            self._cache = []

    def _save(self) -> None:
        """Flush the in-memory cache to the JSONL file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            for entry in self._cache:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # -- Public API ---------------------------------------------------------

    def record(self, suite_name: str, result: SuiteResult, baseline: bool = False) -> str:
        """Persist a suite result and return its run_id.

        Args:
            suite_name: Name of the suite (used for later queries).
            result:     SuiteResult to persist.
            baseline:   If True, mark this run as the baseline for the suite.

        Returns:
            The generated run_id (UUID4 string).
        """
        run_id = str(uuid.uuid4())[:8]
        entry: dict[str, Any] = {
            "run_id": run_id,
            "suite_name": suite_name,
            "result": result.to_dict(),
            "baseline": baseline,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        self._cache.append(entry)
        if baseline:
            self._baselines[suite_name] = run_id
        self._save()
        return run_id

    def get_runs(self, suite_name: str, limit: int | None = None) -> list[SuiteResult]:
        """Return historical runs for a suite, newest first."""
        runs = [
            SuiteResult.from_dict(e["result"])
            for e in reversed(self._cache)
            if e["suite_name"] == suite_name
        ]
        return runs[:limit] if limit else runs

    def get_baseline(self, suite_name: str) -> SuiteResult | None:
        """Return the current baseline run for a suite, or None."""
        baseline_run_id = self._baselines.get(suite_name)
        if not baseline_run_id:
            return None
        for entry in self._cache:
            if entry["run_id"] == baseline_run_id:
                return SuiteResult.from_dict(entry["result"])
        return None

    def set_baseline(self, suite_name: str, run_id: str) -> bool:
        """Mark a specific run as the new baseline. Returns True if found."""
        found = False
        for entry in self._cache:
            if entry["suite_name"] == suite_name and entry["run_id"] == run_id:
                entry["baseline"] = True
                self._baselines[suite_name] = run_id
                found = True
            elif entry["suite_name"] == suite_name:
                entry["baseline"] = False
        if found:
            self._save()
        return found

    def compare(
        self,
        suite_name: str,
        run_a_id: str,
        run_b_id: str,
    ) -> RegressionReport | None:
        """Diff two runs and return a RegressionReport, or None if not found."""
        runs = {e["run_id"]: SuiteResult.from_dict(e["result"]) for e in self._cache if e["suite_name"] == suite_name}
        run_a = runs.get(run_a_id)
        run_b = runs.get(run_b_id)
        if not run_a or not run_b:
            return None

        baseline_score = run_a.avg_score
        current_score = run_b.avg_score
        delta = current_score - baseline_score

        task_scores_a = {r.task_id: r for r in run_a.results}
        task_scores_b = {r.task_id: r for r in run_b.results}

        all_tasks = set(task_scores_a) | set(task_scores_b)
        new_failures: list[str] = []
        improved: list[str] = []
        regressions: list[str] = []

        for tid in all_tasks:
            a = task_scores_a.get(tid)
            b = task_scores_b.get(tid)
            if a is None or b is None:
                continue
            if a.passed and not b.passed:
                new_failures.append(tid)
            if not a.passed and b.passed:
                improved.append(tid)
            if b.score < a.score:
                regressions.append(tid)

        return RegressionReport(
            suite_name=suite_name,
            baseline_run=run_a_id,
            current_run=run_b_id,
            baseline_score=baseline_score,
            current_score=current_score,
            delta_score=delta,
            new_failures=new_failures,
            improved=improved,
            regressions=regressions,
        )

    def check_regression(self, suite_name: str, result: SuiteResult) -> bool:
        """Return True if result scored worse than the suite's current baseline."""
        baseline = self.get_baseline(suite_name)
        if baseline is None:
            return False
        return result.avg_score < baseline.avg_score


# ---------------------------------------------------------------------------
# Agent Runner
# ---------------------------------------------------------------------------


class AgentRunner:
    """Wraps an agent function with timing, cost tracking, and suite support.

    Example:
        runner = AgentRunner(my_agent_fn, cost_fn=lambda task: 0.02)
        result, ms, cost, err = runner.run(task)
        suite_result = runner.run_suite(my_suite)
    """

    def __init__(
        self,
        agent_fn: Callable[[Any], Any],
        cost_fn: Callable[[Any], float] | None = None,
    ) -> None:
        self._agent_fn = agent_fn
        self._cost_fn = cost_fn or (lambda _: 0.0)

    def run(self, task: Any) -> tuple[Any, float, float, str | None]:
        """Run the agent on a single task.

        Returns:
            Tuple of (result, duration_ms, cost, error).
            error is None on success, otherwise a string message.
        """
        error: str | None = None
        duration_ms = 0.0
        cost = 0.0
        result: Any = None
        try:
            start = time.perf_counter()
            result = self._agent_fn(task)
            duration_ms = (time.perf_counter() - start) * 1000
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            result = None
        try:
            cost = self._cost_fn(task)
        except Exception:  # pragma: no cover
            cost = 0.0
        return result, duration_ms, cost, error

    def run_suite(
        self,
        suite: BenchmarkSuite,
        verbose: bool = False,
    ) -> SuiteResult:
        """Run a full benchmark suite using this runner's agent.

        The agent is invoked via `self.run()` rather than directly so that
        timing and cost hooks are applied consistently.
        """
        results: list[EvalResult] = []
        started_at = datetime.now(timezone.utc).isoformat()

        for case in suite._cases.values():
            if verbose:
                print(f"  [{case.task_id}] {case.description}...", end=" ", flush=True)

            task = case.task_fn()
            result, duration_ms, cost, error = self.run(task)

            class _Task:
                pass

            _task = _Task()
            _task.task_id = case.task_id
            if isinstance(task, dict):
                for k, v in task.items():
                    setattr(_task, k, v)

            if error:
                eval_result = EvalResult(
                    task_id=case.task_id,
                    passed=False,
                    score=0.0,
                    criteria_scores=[],
                    duration_ms=duration_ms,
                    cost=cost,
                    error=error,
                )
            else:
                eval_result = case.rubric.evaluate(_task, result)
                eval_result.duration_ms = duration_ms
                eval_result.cost = cost

            results.append(eval_result)

            if verbose:
                status = "PASS" if eval_result.passed else "FAIL"
                print(f"{status} ({eval_result.score:.2f}, {duration_ms:.0f}ms)")

        completed_at = datetime.now(timezone.utc).isoformat()
        total_duration = sum(r.duration_ms for r in results)
        total_cost = sum(r.cost for r in results)
        passed_count = sum(1 for r in results if r.passed)
        avg_score = sum(r.score for r in results) / len(results) if results else 0.0

        return SuiteResult(
            suite_name=suite.name,
            results=results,
            total_duration_ms=total_duration,
            total_cost=total_cost,
            pass_rate=passed_count / len(results) if results else 0.0,
            avg_score=avg_score,
            started_at=started_at,
            completed_at=completed_at,
        )


# ---------------------------------------------------------------------------
# Pre-built Suite Templates
# ---------------------------------------------------------------------------

# -- Helpers ----------------------------------------------------------------


def _compile_score(code_str: str) -> Score:
    """Return Score(1.0) if code_str parses without SyntaxError, else Score(0.0)."""
    try:
        compile(code_str, "<string>", "exec")
        return Score(1.0, "Code compiles successfully")
    except SyntaxError as e:
        return Score(0.0, f"Syntax error: {e.msg} at line {e.lineno}")


def _no_secrets(content: str) -> Score:
    """Return Score(0.0) if common secret patterns are found."""
    patterns = ["api_key", "secret", "password", "token", "aws_access"]
    found = [p for p in patterns if p.lower() in content.lower()]
    if found:
        return Score(0.0, f"Potential secret detected: {', '.join(found)}")
    return Score(1.0, "No obvious secrets found")


def _no_sql_injection(query: str) -> Score:
    """Basic SQL injection check: flag string concatenation in SQL."""
    dangerous = ["'; ", '"', " OR 1=1", " OR '1'='1", "DROP TABLE", "-- "]
    found = [d for d in dangerous if d.lower() in query.lower()]
    if found:
        return Score(0.0, f"Potential SQL injection vector: {found[0]!r}")
    return Score(1.0, "No obvious SQL injection patterns")


def _no_command_injection(cmd: str) -> Score:
    """Basic shell command injection check."""
    dangerous = ["&&", "||", "|", "; rm", ";cat", "`", "$(", "${"]
    found = [d for d in dangerous if d in cmd]
    if found:
        return Score(0.0, f"Potential command injection: {found[0]!r}")
    return Score(1.0, "No obvious command injection patterns")


def _lint_pass(source: str) -> Score:
    """Stub lint check — replace with real ruff/pylint in production."""
    # Always pass; real implementation would call `ruff check` or similar.
    return Score(1.0, "Lint check passed (stub)")


def _error_handling(score: float, reason: str) -> Score:
    """Reusable error-handling criterion factory."""
    return Score(score, reason)


def _has_logging(source: str) -> Score:
    import re
    has_log = bool(re.search(r"\b(log|logger|logging)\b", source, re.IGNORECASE))
    if has_log:
        return Score(1.0, "Logging statements present")
    return Score(0.5, "No logging found (informational — not a hard fail)")


def _has_type_hints(source: str) -> Score:
    import re
    hints = re.findall(r"->\s*\w+|\:\s*(?:int|str|float|bool|list|dict|tuple|None)", source)
    if len(hints) >= 2:
        return Score(1.0, f"Found {len(hints)} type annotations")
    return Score(0.5, "Few or no type hints (informational)")


def _context_fits(token_count: int, limit: int = 128_000) -> Score:
    if token_count <= limit:
        pct = token_count / limit * 100
        return Score(1.0, f"Prompt uses {pct:.1f}% of context window")
    return Score(0.0, f"Exceeds context limit ({token_count} > {limit})")


def _no_redundant_rereading(calls: int, expected: int = 3) -> Score:
    if calls <= expected:
        return Score(1.0, f"Tool calls within budget ({calls} <= {expected})")
    return Score(max(0.0, 1.0 - (calls - expected) * 0.2), f"Excessive tool calls: {calls}")


def _token_budget_respected(used: int, budget: int) -> Score:
    if used <= budget:
        return Score(1.0, f"Token budget respected ({used}/{budget})")
    return Score(0.0, f"Token budget exceeded ({used}/{budget})")


# -- Suite factories --------------------------------------------------------


def bug_fix_suite() -> BenchmarkSuite:
    """Pre-built suite for evaluating bug-fix agents.

    Criteria:
        1. Code compiles / is syntactically valid
        2. Tests pass (stub — replace with real test runner)
        3. No regression in existing tests (stub)
        4. No new hardcoded secrets
        5. No crashes or unhandled exceptions
    """
    suite = BenchmarkSuite(
        "bug-fix",
        description="Evaluates agents on bug-fix tasks: correctness, safety, and stability.",
    )

    cases = [
        BenchmarkCase(
            task_id="compiles",
            description="Bug-fix produces syntactically valid code",
            task_fn=lambda: {},
            expected_fn=lambda: "",
            rubric=Rubric([
                Criterion("syntax", "Output is valid Python syntax", 1.0,
                          lambda t, r: _compile_score(str(r)) if r else Score(0.0, "No output")),
            ]),
        ),
        BenchmarkCase(
            task_id="no_secrets",
            description="Bug-fix does not introduce hardcoded secrets",
            task_fn=lambda: {},
            expected_fn=lambda: "",
            rubric=Rubric([
                Criterion("secrets", "No API keys or secrets in output", 1.0,
                          lambda t, r: _no_secrets(str(r)) if r else Score(1.0, "No output to check")),
            ]),
        ),
        BenchmarkCase(
            task_id="no_crash",
            description="Agent output does not crash on basic inputs",
            task_fn=lambda: {"input": "hello"},
            expected_fn=lambda: "",
            rubric=Rubric([
                Criterion("crash", "No unhandled exceptions", 1.0,
                          lambda t, r: Score(1.0, "No crash detected") if r else Score(0.0, "No output")),
            ]),
        ),
    ]

    for case in cases:
        suite.add_case(case)
    return suite


def code_quality_suite() -> BenchmarkSuite:
    """Pre-built suite for evaluating code quality.

    Criteria:
        1. No syntax errors
        2. Passes lint check (stub)
        3. Proper error handling present
        4. Logging statements present
        5. Type hints present
    """
    suite = BenchmarkSuite(
        "code-quality",
        description="Evaluates code quality: syntax, lint, error handling, logging, types.",
    )

    def _eval_quality(t: Any, r: Any) -> list[Score]:
        src = str(r) if r else ""
        return [
            _compile_score(src),
            _lint_pass(src),
            _error_handling(1.0, "Error handling present"),
            _has_logging(src),
            _has_type_hints(src),
        ]

    case = BenchmarkCase(
        task_id="overall",
        description="Code meets quality standards",
        task_fn=lambda: {},
        expected_fn=lambda: "",
        rubric=Rubric([
            Criterion("syntax", "No syntax errors", 1.0,
                      lambda t, r: _compile_score(str(r) if r else "")),
            Criterion("lint", "Passes lint checks", 1.0,
                      lambda t, r: _lint_pass(str(r) if r else "")),
            Criterion("error_handling", "Has error handling", 1.0,
                      lambda t, r: _error_handling(1.0, "Error handling present")),
            Criterion("logging", "Has logging", 0.5,
                      lambda t, r: _has_logging(str(r) if r else "")),
            Criterion("type_hints", "Has type hints", 0.5,
                      lambda t, r: _has_type_hints(str(r) if r else "")),
        ]),
    )
    suite.add_case(case)
    return suite


def security_suite() -> BenchmarkSuite:
    """Pre-built suite for security-conscious code evaluation.

    Criteria:
        1. No hardcoded secrets
        2. No SQL injection vectors
        3. No command injection vectors
        4. Input validation present
        5. Safe file operations
    """
    suite = BenchmarkSuite(
        "security",
        description="Evaluates security posture: no secrets, no injection, safe I/O.",
    )

    cases = [
        BenchmarkCase(
            task_id="no_hardcoded_secrets",
            description="No API keys, passwords, or tokens in output",
            task_fn=lambda: {},
            expected_fn=lambda: "",
            rubric=Rubric([
                Criterion("secrets", "No hardcoded secrets", 1.0,
                          lambda t, r: _no_secrets(str(r) if r else "")),
            ]),
        ),
        BenchmarkCase(
            task_id="sql_injection",
            description="Output is free of SQL injection vectors",
            task_fn=lambda: {"query": "SELECT * FROM users WHERE id = 1"},
            expected_fn=lambda: "",
            rubric=Rubric([
                Criterion("sql_inj", "No SQL injection", 1.0,
                          lambda t, r: _no_sql_injection(str(r) if r else "")),
            ]),
        ),
        BenchmarkCase(
            task_id="command_injection",
            description="Output is free of command injection vectors",
            task_fn=lambda: {"cmd": "ls"},
            expected_fn=lambda: "",
            rubric=Rubric([
                Criterion("cmd_inj", "No command injection", 1.0,
                          lambda t, r: _no_command_injection(str(r) if r else "")),
            ]),
        ),
        BenchmarkCase(
            task_id="input_validation",
            description="Input validation is present",
            task_fn=lambda: {},
            expected_fn=lambda: "",
            rubric=Rubric([
                Criterion("validation", "Input validation present", 1.0,
                          lambda t, r: Score(1.0, "Validation present") if r else Score(0.5, "No output")),
            ]),
        ),
    ]

    for case in cases:
        suite.add_case(case)
    return suite


def context_efficiency_suite() -> BenchmarkSuite:
    """Pre-built suite for evaluating context-window and token efficiency.

    Criteria:
        1. Prompt fits within context window
        2. No redundant re-reading of files/context
        3. Efficient tool usage
        4. Token budget respected
    """
    suite = BenchmarkSuite(
        "context-efficiency",
        description="Evaluates token efficiency: context usage, redundant calls, tool budget.",
    )

    cases = [
        BenchmarkCase(
            task_id="context_window",
            description="Prompt fits within model context window",
            task_fn=lambda: {"tokens": 4096},
            expected_fn=lambda: 128_000,
            rubric=Rubric([
                Criterion("fits", "Within context window", 1.0,
                          lambda t, r: _context_fits(t.get("tokens", 0))),
            ]),
        ),
        BenchmarkCase(
            task_id="tool_calls",
            description="Agent uses reasonable number of tool calls",
            task_fn=lambda: {"expected_calls": 5},
            expected_fn=lambda: 5,
            rubric=Rubric([
                Criterion("calls", "Tool calls within budget", 1.0,
                          lambda t, r: _no_redundant_rereading(t.get("expected_calls", 0))),
            ]),
        ),
        BenchmarkCase(
            task_id="token_budget",
            description="Token budget is respected",
            task_fn=lambda: {"budget": 10_000},
            expected_fn=lambda: 10_000,
            rubric=Rubric([
                Criterion("budget", "Token budget respected", 1.0,
                          lambda t, r: _token_budget_respected(
                              t.get("actual_tokens", 5000), t.get("budget", 10_000))),
            ]),
        ),
    ]

    for case in cases:
        suite.add_case(case)
    return suite


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

import argparse
import sys


class TinyEvalCLI:
    """Command-line interface for tiny-eval.

    Usage:
        python -m tiny_eval run bug-fix --verbose
        python -m tiny_eval compare run_a.json run_b.json
    """

    def run_suite(self, suite_name_or_path: str, verbose: bool = False) -> SuiteResult:
        """Run a named pre-built suite or load from a Python file."""
        import importlib.util

        if suite_name_or_path in ("bug-fix", "bug_fix"):
            suite = bug_fix_suite()
        elif suite_name_or_path in ("code-quality", "code_quality"):
            suite = code_quality_suite()
        elif suite_name_or_path in ("security",):
            suite = security_suite()
        elif suite_name_or_path in ("context-efficiency", "context_efficiency"):
            suite = context_efficiency_suite()
        else:
            # Try loading as a Python module path
            spec = importlib.util.spec_from_file_location("_suite_mod", suite_name_or_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                suite = mod.suite  # noqa: F821
            else:
                raise ValueError(f"Unknown suite: {suite_name_or_path!r}")

        # Dummy agent that just returns the task
        def dummy_agent(task: Any) -> Any:
            return task

        runner = AgentRunner(dummy_agent)
        result = runner.run_suite(suite, verbose=verbose)
        self.report(result)
        return result

    def compare(self, baseline_path: str, current_path: str) -> RegressionReport:
        """Load two JSON result files and print a comparison."""
        with open(baseline_path, encoding="utf-8") as fh:
            baseline: SuiteResult = SuiteResult.from_dict(json.load(fh))
        with open(current_path, encoding="utf-8") as fh:
            current: SuiteResult = SuiteResult.from_dict(json.load(fh))

        # Build a minimal tracker to call compare()
        tracker = RegressionTracker()
        baseline_id = tracker.record(baseline.suite_name, baseline)
        current_id = tracker.record(current.suite_name, current)
        report = tracker.compare(baseline.suite_name, baseline_id, current_id)

        if report:
            self._print_report(report)
        return report

    def report(self, result: SuiteResult) -> None:
        """Print a formatted SuiteResult to stdout."""
        print("\n" + "=" * 60)
        print(f"  Suite: {result.suite_name}")
        print("=" * 60)
        print(f"  Pass rate : {result.pass_rate:.1%}")
        print(f"  Avg score : {result.avg_score:.3f}")
        print(f"  Cases     : {len(result.results)}")
        print(f"  Duration  : {result.total_duration_ms:.0f} ms")
        print(f"  Cost      : ${result.total_cost:.4f}")
        print("-" * 60)
        for ev in result.results:
            icon = "✅" if ev.passed else "❌"
            err_str = f" [{ev.error}]" if ev.error else ""
            print(f"  {icon} [{ev.task_id}] score={ev.score:.3f}  {ev.duration_ms:.0f}ms{err_str}")
        print("=" * 60 + "\n")

    def _print_report(self, report: RegressionReport) -> None:
        delta = report.delta_score
        emoji = "🔴" if delta < 0 else ("🟢" if delta > 0 else "⚪")
        print(f"\n{emoji} Regression Report: {report.suite_name}")
        print(f"   Baseline : {report.baseline_score:.3f}  (run {report.baseline_run})")
        print(f"   Current  : {report.current_score:.3f}  (run {report.current_run})")
        print(f"   Delta    : {delta:+.3f}")
        if report.regressions:
            print(f"   Regressions: {', '.join(report.regressions)}")
        if report.improved:
            print(f"   Improved : {', '.join(report.improved)}")
        if report.new_failures:
            print(f"   New failures: {', '.join(report.new_failures)}")
        print()


def _main() -> None:
    parser = argparse.ArgumentParser(prog="tiny-eval", description="Zero-dependency agent evaluation")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run an evaluation suite")
    run_p.add_argument("suite", help="Suite name or path to Python file")
    run_p.add_argument("--verbose", "-v", action="store_true")

    cmp_p = sub.add_parser("compare", help="Compare two eval runs")
    cmp_p.add_argument("baseline", help="Path to baseline JSON")
    cmp_p.add_argument("current", help="Path to current JSON")

    args = parser.parse_args()
    cli = TinyEvalCLI()

    if args.command == "run":
        cli.run_suite(args.suite, verbose=args.verbose)
    elif args.command == "compare":
        cli.compare(args.baseline, args.current)
    else:
        parser.print_help()


if __name__ == "__main__":
    _main()
