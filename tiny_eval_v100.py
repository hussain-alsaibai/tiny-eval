"""
tiny-eval — Evaluation framework for autonomous AI agents.

Define benchmark suites, grade agent runs, track regressions, export reports.
Zero dependencies. Built for the autonomous-agent workflow.

MIT License. https://github.com/hussain-alsaibai/tiny-eval
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union


__version__ = "1.0.0"


# ──────────────────────────────────────────────
# Grading Types
# ──────────────────────────────────────────────

class Grade(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


@dataclass
class CriterionResult:
    """Result of evaluating a single criterion."""
    name: str
    grade: Grade
    score: float          # 0.0 – 1.0
    message: str
    details: Optional[dict] = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["grade"] = self.grade.value
        return d


@dataclass
class EvalResult:
    """Result of evaluating a full suite of criteria against one agent run."""
    eval_id: str
    suite_name: str
    agent_name: str
    run_id: str
    timestamp: float
    criteria: list[CriterionResult]
    overall_pass: bool
    overall_score: float   # 0.0 – 1.0
    grade: Grade
    total_duration_ms: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "eval_id": self.eval_id,
            "suite_name": self.suite_name,
            "agent_name": self.agent_name,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "criteria": [c.to_dict() for c in self.criteria],
            "overall_pass": self.overall_pass,
            "overall_score": self.overall_score,
            "grade": self.grade.value,
            "total_duration_ms": self.total_duration_ms,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Human-readable one-line summary."""
        status = "✅ PASS" if self.overall_pass else "❌ FAIL"
        return (f"{status} | {self.suite_name} | {self.agent_name} | "
                f"score={self.overall_score:.1%} | {len(self.criteria)} criteria | "
                f"{self.total_duration_ms:.0f}ms")


@dataclass
class Criterion:
    """
    A single evaluation criterion.

    Args:
        name: Short identifier (e.g., "compiles", "tests_pass", "no_regression")
        description: Human-readable explanation of what this checks
        check: Callable[Dict -> bool | tuple[bool, str]]. Receives run context dict,
               returns True (pass) or False (fail), or (bool, message) for detail.
        weight: Contribution weight (0–1). Defaults to 1.0.
        required: If True, any fail on this criterion makes overall FAIL. Default False.
    """
    name: str
    description: str
    check: Callable[[dict], Union[bool, tuple[bool, str]]]
    weight: float = 1.0
    required: bool = False


# ──────────────────────────────────────────────
# Benchmark Suite
# ──────────────────────────────────────────────

class EvalSuite:
    """
    A named collection of evaluation criteria forming a benchmark.

    Usage:
        suite = EvalSuite("bug-fix-benchmark", [
            Criterion("compiles", "Code must compile without errors",
                      lambda ctx: ctx.get("compile_error") is None),
            Criterion("tests_pass", "All tests pass",
                      lambda ctx: ctx.get("test_failures", 0) == 0, weight=2.0),
            Criterion("no_secrets", "No leaked secrets",
                      lambda ctx: ctx.get("secrets_found", 0) == 0),
        ])
        result = suite.evaluate(
            agent_name="cursor-agent",
            run_id="run-abc123",
            context={"compile_error": None, "test_failures": 0, "secrets_found": 0},
        )
        print(result.summary())
    """

    def __init__(
        self,
        name: str,
        criteria: list[Criterion],
        pass_threshold: float = 0.7,
    ):
        self.name = name
        self.criteria = criteria
        self.pass_threshold = pass_threshold  # overall score to consider PASS

    def evaluate(
        self,
        agent_name: str,
        run_id: str,
        context: dict,
        metadata: Optional[dict] = None,
    ) -> EvalResult:
        """
        Run all criteria against the given context and return an EvalResult.
        """
        start = time.perf_counter()
        results: list[CriterionResult] = []
        weighted_sum = 0.0
        total_weight = 0.0
        any_required_fail = False

        for criterion in self.criteria:
            c_start = time.perf_counter()
            try:
                raw = criterion.check(context)
                if isinstance(raw, tuple):
                    passed, message = raw
                else:
                    passed = bool(raw)
                    message = "passed" if passed else "failed"

            except Exception as exc:
                passed = False
                message = f"criterion raised: {type(exc).__name__}: {exc}"

            c_duration = (time.perf_counter() - c_start) * 1000
            score = 1.0 if passed else 0.0
            grade = Grade.PASS if passed else Grade.FAIL

            if criterion.required and not passed:
                any_required_fail = True

            results.append(CriterionResult(
                name=criterion.name,
                grade=grade,
                score=score,
                message=message,
                details={"description": criterion.description},
                duration_ms=c_duration,
            ))

            weighted_sum += score * criterion.weight
            total_weight += criterion.weight

        duration_ms = (time.perf_counter() - start) * 1000
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        overall_pass = overall_score >= self.pass_threshold and not any_required_fail

        if any_required_fail:
            overall_score = 0.0
            overall_pass = False
            grade = Grade.FAIL
        elif overall_pass:
            grade = Grade.PASS
        else:
            grade = Grade.FAIL

        return EvalResult(
            eval_id=str(uuid.uuid4())[:8],
            suite_name=self.name,
            agent_name=agent_name,
            run_id=run_id,
            timestamp=time.time(),
            criteria=results,
            overall_pass=overall_pass,
            overall_score=overall_score,
            grade=grade,
            total_duration_ms=duration_ms,
            metadata=metadata or {},
        )


# ──────────────────────────────────────────────
# Evaluator — manages suites and reporting
# ──────────────────────────────────────────────

class Evaluator:
    """
    Manages multiple eval suites, runs batches, tracks history.

    Usage:
        ev = Evaluator(storage_path=".tiny-eval")
        ev.register_suite(bug_fix_suite)
        ev.register_suite(security_suite)

        result = ev.run("bug-fix-suit", agent_name="devin", run_id="r1", context={...})
        batch = ev.run_batch("devin", run_contexts=[ctx1, ctx2, ctx3])

        report = ev.generate_report(format="markdown")
        print(report)
    """

    def __init__(self, storage_path: str | None = None):
        self.storage_path = storage_path or os.path.join(
            os.getcwd(), ".tiny-eval"
        )
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        self.suites: dict[str, EvalSuite] = {}

    def register_suite(self, suite: EvalSuite):
        self.suites[suite.name] = suite

    def run(
        self,
        suite_name: str,
        agent_name: str,
        run_id: str,
        context: dict,
        metadata: Optional[dict] = None,
    ) -> EvalResult:
        if suite_name not in self.suites:
            raise KeyError(f"No suite registered with name: {suite_name}. "
                            f"Registered: {list(self.suites.keys())}")
        result = self.suites[suite_name].evaluate(agent_name, run_id, context, metadata)
        self._save_result(result)
        # The index is what list_results() and regression_check() read.
        # Without this line the result is written to disk but never
        # indexed, so listing results always comes back empty and a
        # regression check can only ever say "unknown".
        self._append_index(result)
        return result

    def run_batch(
        self,
        suite_name: str,
        agent_name: str,
        run_contexts: list[dict],
        metadata: Optional[dict] = None,
    ) -> list[EvalResult]:
        """Run a suite against multiple contexts, e.g. multiple agent runs."""
        results = []
        for i, ctx in enumerate(run_contexts):
            run_id = f"batch-{i+1}"
            result = self.run(suite_name, agent_name, run_id, ctx, metadata)
            results.append(result)
        return results

    def _save_result(self, result: EvalResult):
        path = os.path.join(self.storage_path, f"{result.eval_id}.json")
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    def _append_index(self, result: EvalResult):
        idx_path = os.path.join(self.storage_path, ".index.jsonl")
        with open(idx_path, "a") as f:
            f.write(json.dumps({
                "eval_id": result.eval_id,
                "suite_name": result.suite_name,
                "agent_name": result.agent_name,
                "run_id": result.run_id,
                "overall_pass": result.overall_pass,
                "overall_score": result.overall_score,
                "timestamp": result.timestamp,
                "file": f"{result.eval_id}.json",
            }) + "\n")

    def list_results(self, suite_name: str | None = None,
                     agent_name: str | None = None,
                     limit: int = 50) -> list[dict]:
        idx_path = os.path.join(self.storage_path, ".index.jsonl")
        if not os.path.exists(idx_path):
            return []
        results = []
        with open(idx_path) as f:
            for line in f:
                r = json.loads(line)
                if suite_name and r["suite_name"] != suite_name:
                    continue
                if agent_name and r["agent_name"] != agent_name:
                    continue
                results.append(r)
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return results[:limit]

    def regression_check(
        self,
        suite_name: str,
        agent_name: str,
        current_score: float,
        window: int = 5,
        threshold: float = 0.05,
    ) -> dict:
        """
        Check if the current score is a regression vs recent history.

        Returns a dict with:
            is_regression: bool
            previous_avg: float
            delta: float
            trend: "improving" | "stable" | "regressing"
        """
        recent = self.list_results(suite_name, agent_name, limit=window)
        if not recent:
            return {
                "is_regression": False,
                "previous_avg": None,
                "delta": 0.0,
                "trend": "unknown",
            }

        prev_scores = [r["overall_score"] for r in recent]
        prev_avg = sum(prev_scores) / len(prev_scores)
        delta = current_score - prev_avg

        if delta < -threshold:
            trend = "regressing"
            is_regression = True
        elif delta > threshold:
            trend = "improving"
            is_regression = False
        else:
            trend = "stable"
            is_regression = False

        return {
            "is_regression": is_regression,
            "previous_avg": prev_avg,
            "delta": delta,
            "trend": trend,
            "window_size": len(prev_scores),
        }

    def generate_report(
        self,
        suite_name: str | None = None,
        agent_name: str | None = None,
        format: str = "markdown",
        since: float | None = None,
    ) -> str:
        """
        Generate a human-readable report.

        format: "markdown" | "json" | "summary"
        """
        results = self.list_results(suite_name, suite_name, limit=200)
        if since:
            results = [r for r in results if r["timestamp"] >= since]

        if format == "json":
            return json.dumps(results, indent=2, default=str)

        if format == "summary":
            total = len(results)
            passed = sum(1 for r in results if r["overall_pass"])
            avg_score = (sum(r["overall_score"] for r in results) / total) if total else 0
            return (
                f"tiny-eval summary | {total} runs | "
                f"{passed}/{total} passed ({passed/total:.0%}) | "
                f"avg score: {avg_score:.1%}"
            )

        # Markdown
        lines = [
            "# tiny-eval Report",
            "",
            f"**Generated:** {datetime.now().isoformat()}",
            f"**Suite filter:** {suite_name or 'all'}",
            f"**Agent filter:** {agent_name or 'all'}",
            f"**Total runs:** {len(results)}",
        ]

        if results:
            passed = sum(1 for r in results if r["overall_pass"])
            avg = sum(r["overall_score"] for r in results) / len(results)
            lines += [
                f"**Pass rate:** {passed}/{len(results)} ({passed/len(results):.1%})",
                f"**Avg score:** {avg:.1%}",
                "",
                "## Recent Runs",
                "",
                "| Run ID | Agent | Suite | Score | Status | Timestamp |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for r in results[:20]:
                status = "✅" if r["overall_pass"] else "❌"
                ts = datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M")
                lines.append(
                    f"| {r['run_id']} | {r['agent_name']} | "
                    f"{r['suite_name']} | {r['overall_score']:.0%} | "
                    f"{status} | {ts} |"
                )

            lines += ["", "## Trend Analysis", ""]
            by_agent: dict[str, list] = {}
            for r in results:
                by_agent.setdefault(r["agent_name"], []).append(r)

            for agent, runs in sorted(by_agent.items()):
                if len(runs) < 2:
                    continue
                scores = [r["overall_score"] for r in runs]
                avg_s = sum(scores) / len(scores)
                trend_char = "📈" if scores[0] > scores[-1] else "📉"
                lines.append(
                    f"- **{agent}**: {len(runs)} runs | avg {avg_s:.1%} "
                    f"| latest {scores[0]:.1%} {trend_char}"
                )

        return "\n".join(lines)


# ──────────────────────────────────────────────
# Pre-built Standard Suites
# ──────────────────────────────────────────────

def bug_fix_suite() -> EvalSuite:
    """
    Standard benchmark for bug-fix agents.
    Checks: compilation, tests, no new errors, no secrets, code quality.
    """
    return EvalSuite("bug-fix", [
        Criterion(
            "compiles",
            "Code must compile without syntax or import errors",
            lambda ctx: (ctx.get("compile_error") is None, "clean build")
                       if ctx.get("compile_error") is None
                       else (False, f"compile error: {ctx['compile_error']}"),
            required=True,
        ),
        Criterion(
            "tests_pass",
            "All unit tests must pass",
            lambda ctx: (
                ctx.get("test_failures", -1) == 0,
                f"{ctx.get('test_failures', '?')} failures"
                if ctx.get("test_failures", -1) > 0
                else "all pass",
            ),
            weight=2.0,
            required=True,
        ),
        Criterion(
            "no_regression",
            "Existing tests must not regress",
            lambda ctx: ctx.get("regression_count", 0) == 0,
        ),
        Criterion(
            "no_secrets",
            "No secrets or credentials in output",
            lambda ctx: (
                ctx.get("secrets_found", 0) == 0,
                f"{ctx.get('secrets_found', 0)} secrets found"
                if ctx.get("secrets_found", 0) > 0
                else "clean",
            ),
        ),
        Criterion(
            "no_crash",
            "Agent run must not crash",
            lambda ctx: ctx.get("crashed", False) is False,
            required=True,
        ),
        Criterion(
            "reasonable_time",
            "Run should complete in reasonable time (<10min for simple fix)",
            lambda ctx: ctx.get("duration_s", 999) < 600,
        ),
    ], pass_threshold=0.75)


def code_quality_suite() -> EvalSuite:
    """
    Code quality checks: linting, type errors, complexity.
    """
    return EvalSuite("code-quality", [
        Criterion(
            "no_lint_errors",
            "Code must pass linting checks",
            lambda ctx: ctx.get("lint_errors", 999) == 0,
        ),
        Criterion(
            "no_type_errors",
            "Type annotations must be valid (if used)",
            lambda ctx: ctx.get("type_errors", 0) == 0,
        ),
        Criterion(
            "test_coverage",
            "New code should have test coverage",
            lambda ctx: ctx.get("coverage_percent", 0) >= 60,
        ),
        Criterion(
            "no_obvious_bugs",
            "Static analysis should find no obvious bugs",
            lambda ctx: ctx.get("static_bugs", 0) == 0,
        ),
    ], pass_threshold=0.75)


def security_suite() -> EvalSuite:
    """
    Security checks: no secrets, no injection vectors, safe eval.
    """
    return EvalSuite("security", [
        Criterion(
            "no_leaked_secrets",
            "No API keys, tokens, passwords in output",
            lambda ctx: (
                ctx.get("secrets_found", 0) == 0,
                f"{ctx.get('secrets_found', 0)} secrets exposed"
                if ctx.get("secrets_found", 0) > 0
                else "clean",
            ),
            required=True,
        ),
        Criterion(
            "no_code_injection",
            "No eval/exec with unsanitized input",
            lambda ctx: ctx.get("injection_risk", False) is False,
        ),
        Criterion(
            "no_path_traversal",
            "No unsafe file path operations",
            lambda ctx: ctx.get("path_traversal_risk", False) is False,
        ),
    ], pass_threshold=1.0)  # Security suite requires 100%


# ──────────────────────────────────────────────
# Convenience API
# ──────────────────────────────────────────────

_default_evaluator: Optional[Evaluator] = None


def get_evaluator(storage_path: str | None = None) -> Evaluator:
    global _default_evaluator
    if _default_evaluator is None:
        _default_evaluator = Evaluator(storage_path)
        # Register standard suites
        _default_evaluator.register_suite(bug_fix_suite())
        _default_evaluator.register_suite(code_quality_suite())
        _default_evaluator.register_suite(security_suite())
    return _default_evaluator


def eval_run(
    suite_name: str,
    agent_name: str,
    run_id: str,
    context: dict,
    metadata: Optional[dict] = None,
) -> EvalResult:
    """One-shot evaluate a run against a named suite."""
    return get_evaluator().run(suite_name, agent_name, run_id, context, metadata)


def eval_batch(
    suite_name: str,
    agent_name: str,
    run_contexts: list[dict],
) -> list[EvalResult]:
    """Batch evaluate multiple contexts."""
    return get_evaluator().run_batch(suite_name, agent_name, run_contexts)
