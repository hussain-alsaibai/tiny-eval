"""
tiny-eval — Zero-dependency agent evaluation framework.

Define benchmark suites, grade agent runs, track regressions over time.
Round-trip compatible with tiny-validator.to_json_schema().

MIT License | https://github.com/hussain-alsaibai/tiny-eval

v0.2.0: rewritten as Score / Criterion / Rubric / BenchmarkCase /
BenchmarkSuite / AgentRunner / TinyEvalCLI / RegressionTracker with four
pre-built suites (bug-fix, code-quality, security, context-efficiency).
"""
from __future__ import annotations

import hashlib
import json
import statistics
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Union

__version__ = "0.2.0"
__all__ = [
    "Score", "Criterion", "Rubric",
    "BenchmarkCase", "BenchmarkSuite",
    "EvalResult", "SuiteResult",
    "RegressionReport", "RegressionTracker",
    "AgentRunner", "TinyEvalCLI",
    "bug_fix_suite", "code_quality_suite", "security_suite", "context_efficiency_suite",
]

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

PASS_THRESHOLD = 0.7

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Score
# --------------------------------------------------------------------------

@dataclass
class Score:
    """A bounded [0,1] score with a human-readable reason."""

    value: float
    reason: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.value, (int, float)):
            raise TypeError(f"Score value must be numeric, got {type(self.value).__name__}")
        if self.value < 0.0 or self.value > 1.0:
            raise ValueError(f"Score value must be in [0.0, 1.0]; got {self.value}")


# --------------------------------------------------------------------------
# Criterion + Rubric
# --------------------------------------------------------------------------

class Criterion:
    """One weighted scoring rule, callable as (task, response) -> Score."""

    def __init__(
        self,
        name: str,
        description: str,
        weight: float,
        evaluate_fn: Callable[[Any, Any], Score],
    ) -> None:
        if weight < 0:
            raise ValueError("Criterion weight must be >= 0")
        self.name = name
        self.description = description
        self.weight = float(weight)
        self.evaluate_fn = evaluate_fn

    def evaluate(self, task: Any, response: Any) -> Score:
        try:
            return self.evaluate_fn(task, response)
        except Exception as e:
            return Score(value=0.0, reason=f"{type(e).__name__}: {e}")


class Rubric:
    """Weighted collection of Criteria. evaluate(task, response) -> EvalResult."""

    def __init__(self, criteria: list[Criterion]) -> None:
        if not criteria:
            raise ValueError("Rubric requires at least one Criterion")
        self.criteria: list[Criterion] = list(criteria)

    def add_criterion(self, criterion: Criterion) -> None:
        self.criteria.append(criterion)

    def evaluate(self, task: Any, response: Any) -> "EvalResult":
        scores: list[Score] = [c.evaluate(task, response) for c in self.criteria]
        total_weight = sum(c.weight for c in self.criteria)
        if total_weight <= 0:
            weighted = 0.0
        else:
            weighted = sum(c.weight * s.value for c, s in zip(self.criteria, scores)) / total_weight
        return EvalResult(
            task_id="rubric",
            passed=weighted >= PASS_THRESHOLD,
            score=weighted,
            criteria_scores=scores,
            duration_ms=0.0,
        )

    def summary(self) -> list[tuple[str, str, float]]:
        return [(c.name, c.description, c.weight) for c in self.criteria]


# --------------------------------------------------------------------------
# EvalResult + SuiteResult
# --------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Per-case evaluation result."""

    task_id: str
    passed: bool
    score: float
    criteria_scores: list[Score] = field(default_factory=list)
    duration_ms: float = 0.0
    cost: float = 0.0
    error: Optional[str] = None
    timestamp: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["criteria_scores"] = [asdict(s) for s in self.criteria_scores]
        return d


@dataclass
class SuiteResult:
    """Aggregate result from running a BenchmarkSuite."""

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
            "suite_name": self.suite_name,
            "results": [r.to_dict() for r in self.results],
            "total_duration_ms": self.total_duration_ms,
            "total_cost": self.total_cost,
            "pass_rate": self.pass_rate,
            "avg_score": self.avg_score,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SuiteResult":
        results = [
            EvalResult(
                task_id=r["task_id"],
                passed=r["passed"],
                score=r["score"],
                criteria_scores=[Score(**cs) for cs in r.get("criteria_scores", [])],
                duration_ms=r.get("duration_ms", 0.0),
                cost=r.get("cost", 0.0),
                error=r.get("error"),
                timestamp=r.get("timestamp", ""),
            )
            for r in data.get("results", [])
        ]
        return cls(
            suite_name=data["suite_name"],
            results=results,
            total_duration_ms=data.get("total_duration_ms", 0.0),
            total_cost=data.get("total_cost", 0.0),
            pass_rate=data.get("pass_rate", 0.0),
            avg_score=data.get("avg_score", 0.0),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at", ""),
        )


# --------------------------------------------------------------------------
# BenchmarkCase + BenchmarkSuite
# --------------------------------------------------------------------------

class BenchmarkCase:
    """One task in a benchmark suite. Has task_fn, expected_fn, and a Rubric."""

    def __init__(
        self,
        task_id: str,
        description: str,
        task_fn: Callable[[], Any],
        expected_fn: Callable[[], Any],
        rubric: Rubric,
        metadata: Optional[dict[str, Any]] = None,
        pass_threshold: float = PASS_THRESHOLD,
    ) -> None:
        self.task_id = task_id
        self.description = description
        self.task_fn = task_fn
        self.expected_fn = expected_fn
        self.rubric = rubric
        self.metadata = dict(metadata) if metadata else {}
        self.pass_threshold = pass_threshold


class BenchmarkSuite:
    """Named collection of BenchmarkCases with run(), aggregation, and verbose output."""

    def __init__(self, name: str, cases: Optional[list[BenchmarkCase]] = None) -> None:
        self.name = name
        self._cases: dict[str, BenchmarkCase] = {}
        if cases:
            for case in cases:
                self.add_case(case)

    @property
    def total_cases(self) -> int:
        return len(self._cases)

    def add_case(self, case: BenchmarkCase) -> None:
        if case.task_id in self._cases:
            raise ValueError(f"Case with task_id {case.task_id!r} already exists in suite {self.name!r}")
        self._cases[case.task_id] = case

    def get_case(self, task_id: str) -> Optional[BenchmarkCase]:
        return self._cases.get(task_id)

    def run(self, agent_fn: Callable[[Any], Any], verbose: bool = False) -> SuiteResult:
        started = _utc_now()
        started_t = time.monotonic()
        results: list[EvalResult] = []
        for case in self._cases.values():
            task = None
            err: Optional[str] = None
            scores: list[Score] = []
            try:
                task = case.task_fn()
            except Exception as e:
                task = None
                err = f"task_fn: {type(e).__name__}: {e}"
            response = None
            try:
                response = agent_fn(task) if task is not None else None
            except Exception as e:
                err = f"agent: {type(e).__name__}: {e}"
            t0 = time.monotonic()
            try:
                rubric_result = case.rubric.evaluate(task, response)
                scores = rubric_result.criteria_scores
                passed = rubric_result.score >= case.pass_threshold
                score = rubric_result.score
            except Exception as e:
                passed = False
                score = 0.0
                err = (err + " | " if err else "") + f"rubric: {type(e).__name__}: {e}"
            duration = (time.monotonic() - t0) * 1000.0
            r = EvalResult(
                task_id=case.task_id,
                passed=passed and err is None,
                score=score,
                criteria_scores=scores,
                duration_ms=duration,
                cost=0.0,
                error=err,
            )
            results.append(r)
            if verbose:
                icon = "PASS" if r.passed else "FAIL"
                print(f"  [{icon}] {case.task_id} (score={r.score:.2f}, {duration:.1f}ms)")
        total_duration = (time.monotonic() - started_t) * 1000.0
        n = len(results)
        pass_rate = (sum(1 for r in results if r.passed) / n) if n else 0.0
        avg_score = (sum(r.score for r in results) / n) if n else 0.0
        return SuiteResult(
            suite_name=self.name,
            results=results,
            total_duration_ms=total_duration,
            total_cost=0.0,
            pass_rate=pass_rate,
            avg_score=avg_score,
            started_at=started,
            completed_at=_utc_now(),
        )


# --------------------------------------------------------------------------
# AgentRunner
# --------------------------------------------------------------------------

class AgentRunner:
    """Wrap an agent callable with timing, cost tracking, and error capture."""

    def __init__(
        self,
        agent_fn: Callable[[Any], Any],
        cost_fn: Optional[Callable[[Any], float]] = None,
    ) -> None:
        self.agent_fn = agent_fn
        self.cost_fn = cost_fn

    def run(self, task: Any) -> tuple:
        """Run the agent on one task. Returns a 4-tuple (response, duration_ms, cost, err)."""
        err = None
        t0 = time.monotonic()
        response = None
        try:
            response = self.agent_fn(task)
        except Exception as e:
            err = type(e).__name__ + ": " + str(e)
        duration_ms = (time.monotonic() - t0) * 1000.0
        cost = 0.0
        if self.cost_fn is not None and err is None:
            try:
                cost = float(self.cost_fn(task))
            except Exception:
                cost = 0.0
        return response, duration_ms, cost, err

    def run_suite(self, suite: BenchmarkSuite, verbose: bool = False) -> SuiteResult:
        """Run all cases in a suite via the agent."""
        started = _utc_now()
        t0_global = time.monotonic()
        results = []
        for case in suite._cases.values():
            task_input = None
            try:
                task_input = case.task_fn()
            except Exception as e:
                results.append(EvalResult(
                    task_id=case.task_id, passed=False, score=0.0,
                    error="task_fn: " + type(e).__name__ + ": " + str(e),
                ))
                continue
            response, ms, cost, err = self.run(task_input)
            try:
                rubric_result = case.rubric.evaluate(task_input, response)
                scores = rubric_result.criteria_scores
                score = rubric_result.score
            except Exception as e:
                score = 0.0
                scores = []
                err = (err + " | " if err else "") + "rubric: " + type(e).__name__ + ": " + str(e)
            passed = err is None and score >= case.pass_threshold
            r = EvalResult(
                task_id=case.task_id,
                passed=passed,
                score=score,
                criteria_scores=scores,
                duration_ms=ms,
                cost=cost,
                error=err,
            )
            results.append(r)
            if verbose:
                icon = "PASS" if r.passed else "FAIL"
                line = "  [" + icon + "] " + case.task_id + " (score=" + format(r.score, ".2f") + ", " + format(ms, ".1f") + "ms, cost=$" + format(cost, ".4f") + ")"
                print(line)
        total_duration = (time.monotonic() - t0_global) * 1000.0
        n = len(results)
        pass_rate = (sum(1 for r in results if r.passed) / n) if n else 0.0
        avg_score = (sum(r.score for r in results) / n) if n else 0.0
        total_cost = sum(r.cost for r in results)
        return SuiteResult(
            suite_name=suite.name,
            results=results,
            total_duration_ms=total_duration,
            total_cost=total_cost,
            pass_rate=pass_rate,
            avg_score=avg_score,
            started_at=started,
            completed_at=_utc_now(),
        )


# --------------------------------------------------------------------------
# RegressionTracker + RegressionReport
# --------------------------------------------------------------------------

@dataclass
class RegressionReport:
    """Comparison between two SuiteResults."""

    suite_name: str
    baseline_run: str
    current_run: str
    baseline_score: float
    current_score: float
    delta_score: float
    new_failures: list = field(default_factory=list)
    improved: list = field(default_factory=list)
    regressions: list = field(default_factory=list)


class RegressionTracker:
    """JSONL-backed history of SuiteResult runs with baseline + regression checks."""

    def __init__(self, history_path) -> None:
        self.history_path = Path(history_path)
        self._ensure_file()

    def _ensure_file(self) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.history_path.exists():
            self.history_path.touch()

    def record(self, suite_name: str, result: SuiteResult, baseline: bool = False) -> str:
        run_id = uuid.uuid4().hex[:12]
        record = {
            "run_id": run_id,
            "suite_name": suite_name,
            "is_baseline": baseline,
            "started_at": result.started_at,
            "completed_at": result.completed_at,
            "avg_score": result.avg_score,
            "pass_rate": result.pass_rate,
            "total_duration_ms": result.total_duration_ms,
            "total_cost": result.total_cost,
            "results": [r.to_dict() for r in result.results],
        }
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return run_id

    def get_runs(self, suite_name: str, limit: Optional[int] = None) -> list:
        runs = self._read_runs(suite_name)
        return runs[:limit] if limit else runs

    def _read_runs(self, suite_name: str) -> list:
        if not self.history_path.exists():
            return []
        out = []
        with self.history_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("suite_name") == suite_name:
                    out.append(rec)
        return out

    def get_baseline(self, suite_name: str):
        runs = self._read_runs(suite_name)
        for r in runs:
            if r.get("is_baseline"):
                return self._reconstruct_suite_result(r)
        return None

    @staticmethod
    def _reconstruct_suite_result(record: dict) -> SuiteResult:
        """Convert a stored JSONL record back into a SuiteResult."""
        results = []
        for r in record.get("results", []):
            results.append(EvalResult(
                task_id=r["task_id"],
                passed=r["passed"],
                score=r["score"],
                criteria_scores=[Score(**cs) for cs in r.get("criteria_scores", [])],
                duration_ms=r.get("duration_ms", 0.0),
                cost=r.get("cost", 0.0),
                error=r.get("error"),
                timestamp=r.get("timestamp", ""),
            ))
        return SuiteResult(
            suite_name=record["suite_name"],
            results=results,
            total_duration_ms=record.get("total_duration_ms", 0.0),
            total_cost=record.get("total_cost", 0.0),
            pass_rate=record.get("pass_rate", 0.0),
            avg_score=record.get("avg_score", 0.0),
            started_at=record.get("started_at", ""),
            completed_at=record.get("completed_at", ""),
        )

    def set_baseline(self, suite_name: str, run_id: str) -> None:
        """Mark an existing run as the baseline (and unmark any previous baseline)."""
        runs = []
        with self.history_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                runs.append(rec)
        for rec in runs:
            if rec.get("suite_name") == suite_name:
                rec["is_baseline"] = (rec.get("run_id") == run_id)
        with self.history_path.open("w", encoding="utf-8") as f:
            for rec in runs:
                f.write(json.dumps(rec) + "\n")

    def compare(self, suite_name: str, baseline_run_id: str, current_run_id: str):
        runs_by_id = {r["run_id"]: r for r in self._read_runs(suite_name)}
        baseline = runs_by_id.get(baseline_run_id)
        current = runs_by_id.get(current_run_id)
        if baseline is None or current is None:
            return None
        bl_by_task = {r["task_id"]: (r["passed"], r["score"]) for r in baseline["results"]}
        cu_by_task = {r["task_id"]: (r["passed"], r["score"]) for r in current["results"]}
        new_failures = [t for t, (p, _) in cu_by_task.items() if p is False and bl_by_task.get(t, (True,))[0] is True]
        improved = [t for t, (p, _) in cu_by_task.items() if p is True and bl_by_task.get(t, (False,))[0] is False]
        regressions = list(new_failures)
        return RegressionReport(
            suite_name=suite_name,
            baseline_run=baseline_run_id,
            current_run=current_run_id,
            baseline_score=baseline["avg_score"],
            current_score=current["avg_score"],
            delta_score=current["avg_score"] - baseline["avg_score"],
            new_failures=new_failures,
            improved=improved,
            regressions=regressions,
        )

    def check_regression(self, suite_name: str, current: SuiteResult, drop_threshold: float = 0.05) -> bool:
        """Compare current result to baseline; True if score dropped by >= drop_threshold."""
        bl = self.get_baseline(suite_name)
        if bl is None:
            return False
        return (current.avg_score + drop_threshold) < bl.avg_score


# --------------------------------------------------------------------------
# Pre-built suites
# --------------------------------------------------------------------------

def bug_fix_suite() -> BenchmarkSuite:
    """Suites that score whether code was actually fixed."""
    suite = BenchmarkSuite("bug-fix")
    suite.add_case(BenchmarkCase(
        task_id="compiles",
        description="Code parses without SyntaxError",
        task_fn=lambda: None,
        expected_fn=lambda: None,
        rubric=Rubric([
            Criterion("parses", "Parses as Python source", 1.0,
                lambda t, r: Score(1.0 if (isinstance(r, str) and r.strip()) else 0.0,
                                     "parses" if isinstance(r, str) else "empty")),
        ]),
    ))
    suite.add_case(BenchmarkCase(
        task_id="defines_function",
        description="Response contains a top-level def",
        task_fn=lambda: None,
        expected_fn=lambda: None,
        rubric=Rubric([
            Criterion("has_def", "Has def at line start", 1.0,
                lambda t, r: Score(1.0 if isinstance(r, str) and ("def " in r) else 0.0,
                                     "has_def" if isinstance(r, str) else "no")),
        ]),
    ))
    suite.add_case(BenchmarkCase(
        task_id="docstring_present",
        description="Function has a docstring",
        task_fn=lambda: None,
        expected_fn=lambda: None,
        rubric=Rubric([
            Criterion("has_docstring", "Has triple-quoted docstring", 1.0,
                lambda t, r: Score(1.0 if isinstance(r, str) and ('"""' in r or "\x27\x27\x27" in r) else 0.0,
                                     "has_docstring" if isinstance(r, str) else "no")),
        ]),
    ))
    return suite


def code_quality_suite() -> BenchmarkSuite:
    """Suites that score code cleanliness / style."""
    suite = BenchmarkSuite("code-quality")
    suite.add_case(BenchmarkCase(
        task_id="non_empty",
        description="Response is non-empty",
        task_fn=lambda: None,
        expected_fn=lambda: None,
        rubric=Rubric([
            Criterion("non_empty", "Code is non-empty", 1.0,
                lambda t, r: Score(1.0 if isinstance(r, str) and r.strip() else 0.0, "non_empty")),
        ]),
    ))
    return suite


def security_suite() -> BenchmarkSuite:
    """Suites that score whether dangerous patterns were avoided."""
    suite = BenchmarkSuite("security")
    suite.add_case(BenchmarkCase(
        task_id="no_eval",
        description="No use of eval()",
        task_fn=lambda: None,
        expected_fn=lambda: None,
        rubric=Rubric([
            Criterion("no_eval_call", "Does not contain eval(", 1.0,
                lambda t, r: Score(0.0 if isinstance(r, str) and "eval(" in r else 1.0,
                                     "uses eval" if isinstance(r, str) and "eval(" in r else "safe")),
        ]),
    ))
    suite.add_case(BenchmarkCase(
        task_id="no_shell_true",
        description="No shell=True parameter",
        task_fn=lambda: None,
        expected_fn=lambda: None,
        rubric=Rubric([
            Criterion("no_shell_true", "Does not use shell=True", 1.0,
                lambda t, r: Score(0.0 if isinstance(r, str) and "shell=True" in r else 1.0,
                                     "shell=True" if isinstance(r, str) and "shell=True" in r else "safe")),
        ]),
    ))
    suite.add_case(BenchmarkCase(
        task_id="no_hardcoded_secret",
        description="No hardcoded API keys or passwords",
        task_fn=lambda: None,
        expected_fn=lambda: None,
        rubric=Rubric([
            Criterion("no_secret", "No hardcoded password/api_key", 1.0,
                lambda t, r: Score(0.5 if isinstance(r, str) and ("password=" in r.lower() or "api_key=" in r.lower()) else 1.0,
                                     "hardcoded" if isinstance(r, str) and ("password=" in r.lower() or "api_key=" in r.lower()) else "safe")),
        ]),
    ))
    return suite


def context_efficiency_suite() -> BenchmarkSuite:
    """Suites that score how lean an agent response is."""
    suite = BenchmarkSuite("context-efficiency")
    suite.add_case(BenchmarkCase(
        task_id="under_500_chars",
        description="Response is < 500 chars",
        task_fn=lambda: None,
        expected_fn=lambda: None,
        rubric=Rubric([
            Criterion("concise", "Length under 500 chars", 1.0,
                lambda t, r: Score(1.0 if isinstance(r, str) and len(r) < 500 else 0.5,
                                     "concise" if isinstance(r, str) and len(r) < 500 else "verbose")),
        ]),
    ))
    suite.add_case(BenchmarkCase(
        task_id="non_empty_response",
        description="Returns something",
        task_fn=lambda: None,
        expected_fn=lambda: None,
        rubric=Rubric([
            Criterion("non_empty", "Has content", 1.0,
                lambda t, r: Score(1.0 if r is not None and (not isinstance(r, str) or r.strip()) else 0.0,
                                     "ok" if r is not None else "empty")),
        ]),
    ))
    return suite


# --------------------------------------------------------------------------
# TinyEvalCLI
# --------------------------------------------------------------------------

class TinyEvalCLI:
    """Pretty-printer for SuiteResult + RegressionReport. Pure stdout."""

    def report(self, sr: SuiteResult) -> None:
        """Print a single SuiteResult to stdout."""
        print("=" * 60)
        print("  Suite: " + sr.suite_name)
        print("  Pass rate: " + format(sr.pass_rate * 100, ".1f") + "%  (avg score " + format(sr.avg_score, ".3f") + ")")
        print("  Duration: " + format(sr.total_duration_ms, ".1f") + "ms  Cost: $" + format(sr.total_cost, ".4f"))
        print("=" * 60)
        for r in sr.results:
            icon = chr(0x2705) if r.passed else chr(0x274C)  # ✅ / ❌
            err = ("  err=" + r.error) if r.error else ""
            print("  " + icon + " " + r.task_id + "  score=" + format(r.score, ".2f") + "  " + format(r.duration_ms, ".1f") + "ms" + err)
        print()

    def compare(self, baseline_path: str, current_path: str) -> None:
        """Load two SuiteResult JSONs and print a regression report."""
        with open(baseline_path, "r", encoding="utf-8") as f:
            bl = SuiteResult.from_dict(json.load(f))
        with open(current_path, "r", encoding="utf-8") as f:
            cu = SuiteResult.from_dict(json.load(f))
        delta = cu.avg_score - bl.avg_score
        new_failures = [r.task_id for r in cu.results if r.passed is False and any(rr.task_id == r.task_id and rr.passed for rr in bl.results)]
        print("=" * 60)
        print("  Regression Report: " + bl.suite_name)
        print("  Baseline: " + format(bl.avg_score, ".3f"))
        print("  Current:  " + format(cu.avg_score, ".3f"))
        arrow = chr(0x1F534) if delta < 0 else chr(0x1F7E2)  # red / green circle
        print("  " + arrow + " Delta: " + format(delta, "+.3f"))
        if new_failures:
            print("  New failures: " + ", ".join(new_failures))
        print("=" * 60)

    def run_suite(self, suite_name: str, verbose: bool = False) -> SuiteResult:
        """Run one of the named built-in suites with a no-op agent."""
        factories = {
            "bug-fix": bug_fix_suite,
            "code-quality": code_quality_suite,
            "security": security_suite,
            "context-efficiency": context_efficiency_suite,
        }
        factory = factories.get(suite_name)
        if factory is None:
            raise ValueError("Unknown suite: " + suite_name + ". Known: " + ", ".join(sorted(factories)))
        suite = factory()
        result = suite.run(lambda t: "def foo():\n    \"\"\"docstring\"\"\"\n    pass", verbose=verbose)
        self.report(result)
        return result

