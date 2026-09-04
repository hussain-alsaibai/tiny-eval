"""
tiny-eval — Zero-dependency agent evaluation framework.

Define benchmark suites, grade agent runs, track regressions over time.
MIT License | https://github.com/hussain-alsaibai/tiny-eval
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

__version__ = "0.1.0"
__all__ = [
    "EvalSuite",
    "EvalCase",
    "EvalResult",
    "EvalRun",
    "RegressionTracker",
    "evaluate",
]


# ──────────────────────────────────────────────────────────────────────────────
# Core data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    """A single test case in an evaluation suite."""

    id: str
    description: str
    input_fn: Callable[[], Any]
    expected_fn: Callable[[Any], float]
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def run(self) -> tuple[Any, float]:
        """Execute the case and return (output, score)."""
        output = self.input_fn()
        score = self.expected_fn(output)
        return output, score


@dataclass
class EvalResult:
    """Result of a single eval case."""

    case_id: str
    case_description: str
    passed: bool
    score: float
    threshold: float
    duration_ms: float
    output: Any
    error: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class EvalRun:
    """Result of a full suite run."""

    id: str
    suite_name: str
    suite_version: str
    timestamp: str
    duration_ms: float
    total_cases: int
    passed_cases: int
    failed_cases: int
    total_score: float
    max_score: float
    pass_rate: float
    results: list[EvalResult]
    agent_info: dict[str, Any] = field(default_factory=dict)
    regression_delta: Optional[float] = None

    def passed(self) -> bool:
        return self.failed_cases == 0

    def summary(self) -> str:
        pct = self.pass_rate * 100
        status = "✅ PASS" if self.passed() else "❌ FAIL"
        return (
            f"{status} | {self.suite_name}@{self.suite_version} | "
            f"{self.passed_cases}/{self.total_cases} cases | "
            f"{pct:.0f}% | score {self.total_score:.1f}/{self.max_score:.1f} | "
            f"{self.duration_ms:.0f}ms"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Suite definition
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalSuite:
    """
    A collection of eval cases with shared metadata and threshold.

    Example:
        suite = EvalSuite(
            name="bug-fix",
            version="1.0.0",
            threshold=0.8,
            cases=[
                EvalCase("no_crash", "Agent must not crash",
                    lambda: agent.run("fix this bug"),
                    lambda out: 1.0 if out and not out.get("crashed") else 0.0,
                    tags=["safety"],
                ),
            ],
        )
    """

    name: str
    version: str
    cases: list[EvalCase]
    description: str = ""
    threshold: float = 0.7
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def run(
        self,
        agent_fn: Callable[[str], Any],
        agent_info: Optional[dict[str, Any]] = None,
    ) -> EvalRun:
        """
        Run all cases against an agent function.

        Args:
            agent_fn: Function that takes a prompt/case_id and returns agent output.
            agent_info: Metadata about the agent (model, version, etc.)

        Returns:
            EvalRun with per-case results and aggregate scores.
        """
        run_id = str(uuid.uuid4())[:8]
        start = time.time()
        results: list[EvalResult] = []
        passed = 0
        failed = 0
        total_score = 0.0

        for case in self.cases:
            case_start = time.time()
            try:
                # The agent receives the case's input prompt
                output = agent_fn(case.description)
                score = case.expected_fn(output)
                duration = (time.time() - case_start) * 1000
                p = score >= self.threshold
                results.append(
                    EvalResult(
                        case_id=case.id,
                        case_description=case.description,
                        passed=p,
                        score=score,
                        threshold=self.threshold,
                        duration_ms=duration,
                        output=output,
                        tags=case.tags,
                    )
                )
                if p:
                    passed += 1
                else:
                    failed += 1
                total_score += score
            except Exception as exc:
                duration = (time.time() - case_start) * 1000
                results.append(
                    EvalResult(
                        case_id=case.id,
                        case_description=case.description,
                        passed=False,
                        score=0.0,
                        threshold=self.threshold,
                        duration_ms=duration,
                        output=None,
                        error=f"{type(exc).__name__}: {exc}",
                        tags=case.tags,
                    )
                )
                failed += 1

        duration_ms = (time.time() - start) * 1000
        total_cases = len(self.cases)
        max_score = total_cases
        pass_rate = passed / total_cases if total_cases else 0.0

        return EvalRun(
            id=run_id,
            suite_name=self.name,
            suite_version=self.version,
            timestamp=datetime.utcnow().isoformat() + "Z",
            duration_ms=duration_ms,
            total_cases=total_cases,
            passed_cases=passed,
            failed_cases=failed,
            total_score=total_score,
            max_score=max_score,
            pass_rate=pass_rate,
            results=results,
            agent_info=agent_info or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "threshold": self.threshold,
            "tags": self.tags,
            "case_count": len(self.cases),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Standalone evaluate() helper
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    agent_fn: Callable[[str], Any],
    cases: list[tuple[str, str, Callable[[Any], float]]],
    *,
    suite_name: str = "adhoc",
    threshold: float = 0.7,
    agent_info: Optional[dict[str, Any]] = None,
) -> EvalRun:
    """
    Quick eval without a full suite definition.

    Args:
        agent_fn: Function(prompt) -> output.
        cases: List of (case_id, description, scoring_fn).
        threshold: Minimum score to count as passed.
        agent_info: Agent metadata dict.

    Returns:
        EvalRun with results.

    Example:
        result = evaluate(
            my_agent,
            [
                ("no_crash", "Must not crash on valid input",
                    lambda o: 1.0 if o else 0.0),
                ("has_output", "Must return non-empty result",
                    lambda o: 1.0 if o and str(o).strip() else 0.0),
            ],
        )
    """
    eval_cases = [
        EvalCase(case_id, desc, lambda d=desc: agent_fn(d), scorer, tags=[])
        for case_id, desc, scorer in cases
    ]
    suite = EvalSuite(
        name=suite_name,
        version="0.0.0",
        cases=eval_cases,
        threshold=threshold,
    )
    return suite.run(agent_fn, agent_info)


# ──────────────────────────────────────────────────────────────────────────────
# Regression tracker
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RegressionRecord:
    run_id: str
    suite_name: str
    suite_version: str
    timestamp: str
    total_score: float
    pass_rate: float
    passed: bool
    agent_info: dict[str, Any]


class RegressionTracker:
    """
    Track eval history and detect regressions.

    Stores runs in a local JSON file. On each new run, compares against
    the previous best score. Optionally blocks on regression.

    Example:
        tracker = RegressionTracker("runs.json")

        run = suite.run(my_agent, {"model": "gpt-4o"})
        result = tracker.record(run)
        if result.regression:
            print(f"⚠️  Regression detected: {result.delta:.3f}")
    """

    def __init__(self, path: str = "eval_runs.json"):
        self.path = Path(path)
        self.runs: list[RegressionRecord] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.runs = [RegressionRecord(**r) for r in data]
            except Exception:
                self.runs = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(r) for r in self.runs], indent=2))

    def record(self, run: EvalRun) -> RegressionResult:
        """
        Record a run and check for regression vs. the last run of the same suite.
        """
        record = RegressionRecord(
            run_id=run.id,
            suite_name=run.suite_name,
            suite_version=run.suite_version,
            timestamp=run.timestamp,
            total_score=run.total_score,
            pass_rate=run.pass_rate,
            passed=run.passed(),
            agent_info=run.agent_info,
        )
        self.runs.append(record)
        self._save()

        # Find previous run for this suite
        prev = None
        for r in reversed(self.runs[:-1]):
            if r.suite_name == run.suite_name:
                prev = r
                break

        delta = None
        regression = False
        if prev is not None:
            delta = run.total_score - prev.total_score
            regression = delta < -0.01  # 1% tolerance

        return RegressionResult(
            record=record,
            previous_record=prev,
            delta=delta,
            regression=regression,
        )

    def best(self, suite_name: str) -> Optional[RegressionRecord]:
        """Return the best-scoring run for a suite."""
        candidates = [r for r in self.runs if r.suite_name == suite_name]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.total_score)

    def history(self, suite_name: str) -> list[RegressionRecord]:
        """Return all runs for a suite, newest first."""
        runs = [r for r in self.runs if r.suite_name == suite_name]
        return sorted(runs, key=lambda r: r.timestamp, reverse=True)


@dataclass
class RegressionResult:
    record: RegressionRecord
    previous_record: Optional[RegressionRecord]
    delta: Optional[float]
    regression: bool

    def summary(self) -> str:
        if self.previous_record is None:
            return f"First run: {self.record.total_score:.1f} (no baseline)"
        arrow = "📈" if self.delta and self.delta > 0 else "📉" if self.regression else "➡️"
        delta_str = f"{self.delta:+.1f}" if self.delta is not None else "?"
        return f"{arrow} {self.record.suite_name} {delta_str} (was {self.previous_record.total_score:.1f})"


# ──────────────────────────────────────────────────────────────────────────────
# CLI helpers
# ──────────────────────────────────────────────────────────────────────────────

def print_run_summary(run: EvalRun) -> None:
    print(f"\n{'='*60}")
    print(f"  {run.summary()}")
    print(f"{'='*60}")
    for r in run.results:
        icon = "✅" if r.passed else "❌"
        err_str = f" | ERROR: {r.error[:60]}" if r.error else ""
        print(f"  {icon} [{r.case_id}] {r.case_description}")
        print(f"       score={r.score:.2f} (threshold={r.threshold}) | {r.duration_ms:.0f}ms{err_str}")
    if run.regression_delta is not None:
        arrow = "📈" if run.regression_delta > 0 else "📉"
        print(f"\n  {arrow} vs previous: {run.regression_delta:+.2f}")
    print()
