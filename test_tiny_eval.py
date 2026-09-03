"""
test_tiny_eval.py — Comprehensive test suite for tiny-eval.
Run with: pytest test_tiny_eval.py -v
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path

import pytest

from tiny_eval import (
    AgentRunner,
    BenchmarkCase,
    BenchmarkSuite,
    Criterion,
    EvalResult,
    RegressionReport,
    RegressionTracker,
    Rubric,
    Score,
    SuiteResult,
    TinyEvalCLI,
    bug_fix_suite,
    code_quality_suite,
    context_efficiency_suite,
    security_suite,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_path():
    """Provide a temporary JSONL path that auto-cleans."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def score_always_pass():
    return Score(value=1.0, reason="always pass")


@pytest.fixture
def score_always_fail():
    return Score(value=0.0, reason="always fail")


@pytest.fixture
def always_pass_criterion(score_always_pass):
    return Criterion(
        name="always_pass",
        description="Always passes",
        weight=1.0,
        evaluate_fn=lambda t, r: score_always_pass,
    )


@pytest.fixture
def always_fail_criterion(score_always_fail):
    return Criterion(
        name="always_fail",
        description="Always fails",
        weight=1.0,
        evaluate_fn=lambda t, r: score_always_fail,
    )


@pytest.fixture
def pass_rubric(always_pass_criterion):
    return Rubric([always_pass_criterion])


@pytest.fixture
def fail_rubric(always_fail_criterion):
    return Rubric([always_fail_criterion])


@pytest.fixture
def pass_case(pass_rubric):
    return BenchmarkCase(
        task_id="pass_case",
        description="Always passes",
        task_fn=lambda: {},
        expected_fn=lambda: None,
        rubric=pass_rubric,
    )


@pytest.fixture
def fail_case(fail_rubric):
    return BenchmarkCase(
        task_id="fail_case",
        description="Always fails",
        task_fn=lambda: {},
        expected_fn=lambda: None,
        rubric=fail_rubric,
    )


# ---------------------------------------------------------------------------
# Score dataclass
# ---------------------------------------------------------------------------

class TestScore:
    def test_score_valid(self):
        s = Score(value=0.75, reason="good")
        assert s.value == 0.75
        assert s.reason == "good"
        assert s.metadata == {}

    def test_score_with_metadata(self):
        s = Score(value=0.5, reason="ok", metadata={"raw": 42})
        assert s.metadata["raw"] == 42

    def test_score_invalid_negative(self):
        with pytest.raises(ValueError, match="0.0, 1.0"):
            Score(value=-0.1, reason="bad")

    def test_score_invalid_over(self):
        with pytest.raises(ValueError, match="0.0, 1.0"):
            Score(value=1.5, reason="bad")

    def test_score_boundary_zero(self):
        s = Score(value=0.0, reason="zero")
        assert s.value == 0.0

    def test_score_boundary_one(self):
        s = Score(value=1.0, reason="one")
        assert s.value == 1.0


# ---------------------------------------------------------------------------
# Criterion & Rubric
# ---------------------------------------------------------------------------

class TestCriterion:
    def test_criterion_valid(self, score_always_pass):
        c = Criterion("acc", "Accuracy", 2.0, lambda t, r: score_always_pass)
        assert c.name == "acc"
        assert c.weight == 2.0

    def test_criterion_negative_weight(self):
        with pytest.raises(ValueError):
            Criterion("bad", "Bad", -1.0, lambda t, r: Score(1.0, "ok"))


class TestRubric:
    def test_rubric_single_criterion(self, always_pass_criterion):
        rubric = Rubric([always_pass_criterion])
        result = rubric.evaluate({}, None)
        assert isinstance(result, EvalResult)
        assert result.passed is True
        assert result.score == 1.0

    def test_rubric_multiple_criteria(self):
        rubric = Rubric([
            Criterion("a", "A", 1.0, lambda t, r: Score(1.0, "a")),
            Criterion("b", "B", 1.0, lambda t, r: Score(0.5, "b")),
        ])
        result = rubric.evaluate({}, None)
        assert result.score == 0.75

    def test_rubric_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            Rubric([])

    def test_rubric_add_criterion(self, pass_rubric):
        new_c = Criterion("added", "Added", 1.0, lambda t, r: Score(1.0, "ok"))
        pass_rubric.add_criterion(new_c)
        assert len(pass_rubric.criteria) == 2

    def test_rubric_summary(self, pass_rubric):
        summary = pass_rubric.summary()
        assert len(summary) == 1
        name, desc, weight = summary[0]
        assert name == "always_pass"
        assert weight == 1.0

    def test_rubric_criterion_raises(self):
        rubric = Rubric([
            Criterion("crash", "Crashes", 1.0, lambda t, r: (_ for _ in ()).throw(RuntimeError("boom")))
        ])
        result = rubric.evaluate({}, None)
        assert result.score == 0.0
        assert "RuntimeError" in result.criteria_scores[0].reason

    def test_rubric_weighted(self):
        rubric = Rubric([
            Criterion("a", "A", 3.0, lambda t, r: Score(1.0, "a")),
            Criterion("b", "B", 1.0, lambda t, r: Score(0.0, "b")),
        ])
        result = rubric.evaluate({}, None)
        # (3*1 + 1*0) / 4 = 0.75
        assert result.score == 0.75


# ---------------------------------------------------------------------------
# BenchmarkCase & BenchmarkSuite
# ---------------------------------------------------------------------------

class TestBenchmarkCase:
    def test_case_creation(self, pass_case):
        assert pass_case.task_id == "pass_case"
        assert pass_case.description == "Always passes"
        assert pass_case.rubric is not None

    def test_case_metadata(self):
        case = BenchmarkCase(
            task_id="m",
            description="meta",
            task_fn=lambda: {},
            expected_fn=lambda: None,
            rubric=Rubric([Criterion("x", "X", 1.0, lambda t, r: Score(1.0, "ok"))]),
            metadata={"tag": "fast"},
        )
        assert case.metadata["tag"] == "fast"


class TestBenchmarkSuite:
    def test_suite_empty(self):
        suite = BenchmarkSuite("empty")
        assert suite.total_cases == 0
        assert suite.name == "empty"

    def test_suite_add_case(self, pass_case):
        suite = BenchmarkSuite("test")
        suite.add_case(pass_case)
        assert suite.total_cases == 1
        assert suite.get_case("pass_case") is pass_case

    def test_suite_add_duplicate_raises(self, pass_case):
        suite = BenchmarkSuite("test")
        suite.add_case(pass_case)
        with pytest.raises(ValueError, match="already exists"):
            suite.add_case(pass_case)

    def test_suite_run_all_pass(self, pass_case):
        suite = BenchmarkSuite("run-test", [pass_case])
        result = suite.run(lambda t: "any")
        assert result.suite_name == "run-test"
        assert len(result.results) == 1
        assert result.results[0].passed is True
        assert result.pass_rate == 1.0
        assert result.avg_score == 1.0
        assert result.total_duration_ms > 0

    def test_suite_run_all_fail(self, fail_case):
        suite = BenchmarkSuite("run-test", [fail_case])
        result = suite.run(lambda t: "any")
        assert result.results[0].passed is False
        assert result.pass_rate == 0.0

    def test_suite_run_verbose(self, pass_case, capsys):
        suite = BenchmarkSuite("v", [pass_case])
        suite.run(lambda t: "ok", verbose=True)
        captured = capsys.readouterr()
        assert "PASS" in captured.out

    def test_suite_run_agent_error(self, pass_rubric):
        case = BenchmarkCase(
            task_id="crash",
            description="crashes",
            task_fn=lambda: {},
            expected_fn=lambda: None,
            rubric=pass_rubric,
        )
        suite = BenchmarkSuite("err", [case])
        result = suite.run(lambda t: (_ for _ in ()).throw(RuntimeError("boom")))
        assert result.results[0].error is not None
        assert "RuntimeError" in result.results[0].error
        assert result.results[0].passed is False

    def test_suite_aggregations(self, pass_case, fail_case):
        suite = BenchmarkSuite("agg", [pass_case, fail_case])
        result = suite.run(lambda t: "ok")
        assert result.pass_rate == 0.5
        assert result.avg_score == 0.5
        assert len(result.results) == 2

    def test_suite_get_case_missing(self, pass_case):
        suite = BenchmarkSuite("x", [pass_case])
        assert suite.get_case("nonexistent") is None


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------

class TestEvalResult:
    def test_eval_result_fields(self):
        ev = EvalResult(
            task_id="t1",
            passed=True,
            score=0.85,
            criteria_scores=[Score(0.85, "ok")],
            duration_ms=123.4,
            cost=0.05,
        )
        assert ev.task_id == "t1"
        assert ev.passed is True
        assert ev.score == 0.85
        assert ev.error is None

    def test_eval_result_to_dict(self):
        ev = EvalResult(
            task_id="t1",
            passed=True,
            score=0.85,
            criteria_scores=[],
            duration_ms=0,
        )
        d = ev.to_dict()
        assert d["task_id"] == "t1"
        assert "timestamp" in d


# ---------------------------------------------------------------------------
# SuiteResult
# ---------------------------------------------------------------------------

class TestSuiteResult:
    def test_suite_result_fields(self):
        sr = SuiteResult(
            suite_name="test",
            results=[],
            total_duration_ms=100.0,
            total_cost=0.0,
            pass_rate=0.0,
            avg_score=0.0,
            started_at="2024-01-01T00:00:00Z",
            completed_at="2024-01-01T00:00:01Z",
        )
        assert sr.suite_name == "test"
        assert len(sr.results) == 0

    def test_suite_result_roundtrip(self):
        sr = SuiteResult(
            suite_name="roundtrip",
            results=[
                EvalResult(
                    task_id="a",
                    passed=True,
                    score=1.0,
                    criteria_scores=[],
                    duration_ms=10.0,
                )
            ],
            total_duration_ms=10.0,
            total_cost=0.0,
            pass_rate=1.0,
            avg_score=1.0,
            started_at="2024-01-01T00:00:00Z",
            completed_at="2024-01-01T00:00:01Z",
        )
        data = sr.to_dict()
        restored = SuiteResult.from_dict(data)
        assert restored.suite_name == "roundtrip"
        assert len(restored.results) == 1
        assert restored.results[0].task_id == "a"


# ---------------------------------------------------------------------------
# RegressionTracker
# ---------------------------------------------------------------------------

class TestRegressionTracker:
    def test_record_and_retrieve(self, temp_path):
        tracker = RegressionTracker(temp_path)
        sr = SuiteResult(
            suite_name="math",
            results=[],
            total_duration_ms=100,
            total_cost=0,
            pass_rate=1.0,
            avg_score=1.0,
            started_at="2024-01-01T00:00:00Z",
            completed_at="2024-01-01T00:00:01Z",
        )
        run_id = tracker.record("math", sr)
        assert run_id is not None
        runs = tracker.get_runs("math")
        assert len(runs) == 1

    def test_baseline(self, temp_path):
        tracker = RegressionTracker(temp_path)
        sr = SuiteResult(
            suite_name="math",
            results=[],
            total_duration_ms=0,
            total_cost=0,
            pass_rate=1.0,
            avg_score=1.0,
            started_at="",
            completed_at="",
        )
        run_id = tracker.record("math", sr, baseline=True)
        baseline = tracker.get_baseline("math")
        assert baseline is not None
        assert baseline.avg_score == 1.0

    def test_set_baseline(self, temp_path):
        tracker = RegressionTracker(temp_path)
        sr = SuiteResult(
            suite_name="math",
            results=[],
            total_duration_ms=0,
            total_cost=0,
            pass_rate=0.5,
            avg_score=0.5,
            started_at="",
            completed_at="",
        )
        run1 = tracker.record("math", sr)
        sr2 = SuiteResult(
            suite_name="math",
            results=[],
            total_duration_ms=0,
            total_cost=0,
            pass_rate=1.0,
            avg_score=1.0,
            started_at="",
            completed_at="",
        )
        run2 = tracker.record("math", sr2)
        tracker.set_baseline("math", run1)
        baseline = tracker.get_baseline("math")
        assert baseline is not None
        assert baseline.avg_score == 0.5

    def test_compare(self, temp_path):
        tracker = RegressionTracker(temp_path)
        sr_a = SuiteResult(
            suite_name="math",
            results=[
                EvalResult("t1", True, 1.0, [], 0),
                EvalResult("t2", True, 1.0, [], 0),
            ],
            total_duration_ms=0,
            total_cost=0,
            pass_rate=1.0,
            avg_score=1.0,
            started_at="",
            completed_at="",
        )
        sr_b = SuiteResult(
            suite_name="math",
            results=[
                EvalResult("t1", False, 0.3, [], 0),
                EvalResult("t2", True, 1.0, [], 0),
            ],
            total_duration_ms=0,
            total_cost=0,
            pass_rate=0.5,
            avg_score=0.65,
            started_at="",
            completed_at="",
        )
        id_a = tracker.record("math", sr_a)
        id_b = tracker.record("math", sr_b)
        report = tracker.compare("math", id_a, id_b)
        assert report is not None
        assert report.delta_score == -0.35
        assert "t1" in report.regressions
        assert "t1" in report.new_failures

    def test_check_regression_no_baseline(self, temp_path):
        tracker = RegressionTracker(temp_path)
        sr = SuiteResult(
            suite_name="x",
            results=[],
            total_duration_ms=0,
            total_cost=0,
            pass_rate=0.5,
            avg_score=0.5,
            started_at="",
            completed_at="",
        )
        # No baseline set — should not flag regression
        assert tracker.check_regression("x", sr) is False

    def test_check_regression_detected(self, temp_path):
        tracker = RegressionTracker(temp_path)
        sr_good = SuiteResult(
            suite_name="x",
            results=[],
            total_duration_ms=0,
            total_cost=0,
            pass_rate=1.0,
            avg_score=1.0,
            started_at="",
            completed_at="",
        )
        sr_bad = SuiteResult(
            suite_name="x",
            results=[],
            total_duration_ms=0,
            total_cost=0,
            pass_rate=0.0,
            avg_score=0.0,
            started_at="",
            completed_at="",
        )
        tracker.record("x", sr_good, baseline=True)
        assert tracker.check_regression("x", sr_bad) is True

    def test_check_regression_no_regression(self, temp_path):
        tracker = RegressionTracker(temp_path)
        sr_good = SuiteResult(
            suite_name="x",
            results=[],
            total_duration_ms=0,
            total_cost=0,
            pass_rate=1.0,
            avg_score=1.0,
            started_at="",
            completed_at="",
        )
        sr_better = SuiteResult(
            suite_name="x",
            results=[],
            total_duration_ms=0,
            total_cost=0,
            pass_rate=1.0,
            avg_score=1.0,
            started_at="",
            completed_at="",
        )
        tracker.record("x", sr_good, baseline=True)
        assert tracker.check_regression("x", sr_better) is False

    def test_compare_missing_run(self, temp_path):
        tracker = RegressionTracker(temp_path)
        sr = SuiteResult(
            suite_name="x",
            results=[],
            total_duration_ms=0,
            total_cost=0,
            pass_rate=1.0,
            avg_score=1.0,
            started_at="",
            completed_at="",
        )
        run_id = tracker.record("x", sr)
        assert tracker.compare("x", run_id, "nonexistent") is None

    def test_get_runs_limit(self, temp_path):
        tracker = RegressionTracker(temp_path)
        for i in range(5):
            sr = SuiteResult(
                suite_name="x",
                results=[],
                total_duration_ms=0,
                total_cost=0,
                pass_rate=1.0,
                avg_score=1.0,
                started_at="",
                completed_at="",
            )
            tracker.record("x", sr)
        runs = tracker.get_runs("x", limit=3)
        assert len(runs) == 3


# ---------------------------------------------------------------------------
# Pre-built Suites
# ---------------------------------------------------------------------------

class TestPrebuiltSuites:
    def test_bug_fix_suite(self):
        suite = bug_fix_suite()
        assert suite.name == "bug-fix"
        assert suite.total_cases >= 3

    def test_code_quality_suite(self):
        suite = code_quality_suite()
        assert suite.name == "code-quality"
        assert suite.total_cases >= 1

    def test_security_suite(self):
        suite = security_suite()
        assert suite.name == "security"
        assert suite.total_cases >= 3

    def test_context_efficiency_suite(self):
        suite = context_efficiency_suite()
        assert suite.name == "context-efficiency"
        assert suite.total_cases >= 2

    def test_bug_fix_suite_runs(self):
        suite = bug_fix_suite()
        result = suite.run(lambda t: "def foo():\n    pass")
        assert result.suite_name == "bug-fix"
        assert isinstance(result.pass_rate, float)
        assert isinstance(result.avg_score, float)

    def test_code_quality_suite_runs(self):
        suite = code_quality_suite()
        result = suite.run(lambda t: "def foo():\n    pass")
        assert result.suite_name == "code-quality"

    def test_security_suite_runs(self):
        suite = security_suite()
        result = suite.run(lambda t: "SELECT * FROM users")
        assert result.suite_name == "security"

    def test_context_efficiency_suite_runs(self):
        suite = context_efficiency_suite()
        result = suite.run(lambda t: None)
        assert result.suite_name == "context-efficiency"


# ---------------------------------------------------------------------------
# AgentRunner
# ---------------------------------------------------------------------------

class TestAgentRunner:
    def test_runner_run_success(self):
        runner = AgentRunner(lambda t: t["x"] * 2)
        result, ms, cost, err = runner.run({"x": 3})
        assert result == 6
        assert ms > 0
        assert err is None

    def test_runner_run_with_cost(self):
        runner = AgentRunner(lambda t: t, cost_fn=lambda t: 0.05)
        result, ms, cost, err = runner.run({})
        assert cost == 0.05

    def test_runner_run_error(self):
        runner = AgentRunner(lambda t: (_ for _ in ()).throw(RuntimeError("oops")))
        result, ms, cost, err = runner.run({})
        assert result is None
        assert err is not None
        assert "RuntimeError" in err

    def test_runner_run_suite(self, pass_case):
        suite = BenchmarkSuite("runner-test", [pass_case])
        runner = AgentRunner(lambda t: "ok")
        result = runner.run_suite(suite)
        assert result.suite_name == "runner-test"
        assert result.results[0].passed is True

    def test_runner_run_suite_verbose(self, pass_case, capsys):
        suite = BenchmarkSuite("v", [pass_case])
        runner = AgentRunner(lambda t: "ok")
        runner.run_suite(suite, verbose=True)
        captured = capsys.readouterr()
        assert "PASS" in captured.out


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_suite(self):
        suite = BenchmarkSuite("empty")
        result = suite.run(lambda t: None)
        assert result.pass_rate == 0.0
        assert result.avg_score == 0.0
        assert len(result.results) == 0

    def test_rubric_all_zero_weights(self):
        rubric = Rubric([
            Criterion("a", "A", 0.0, lambda t, r: Score(1.0, "a")),
        ])
        result = rubric.evaluate({}, None)
        assert result.score == 0.0

    def test_eval_result_threshold(self):
        # Score exactly 0.7 should pass
        rubric = Rubric([
            Criterion("x", "X", 1.0, lambda t, r: Score(0.7, "borderline")),
        ])
        result = rubric.evaluate({}, None)
        assert result.passed is True

        # Score 0.699 should fail
        rubric2 = Rubric([
            Criterion("x", "X", 1.0, lambda t, r: Score(0.699, "just below")),
        ])
        result2 = rubric2.evaluate({}, None)
        assert result2.passed is False

    def test_missing_baseline_returns_none(self, temp_path):
        tracker = RegressionTracker(temp_path)
        assert tracker.get_baseline("never-seen") is None

    def test_regression_report_fields(self):
        report = RegressionReport(
            suite_name="x",
            baseline_run="b",
            current_run="c",
            baseline_score=1.0,
            current_score=0.5,
            delta_score=-0.5,
            new_failures=["t1"],
            improved=[],
            regressions=["t1"],
        )
        assert report.delta_score == -0.5
        assert "t1" in report.new_failures

    def test_suite_result_json_serde(self):
        sr = SuiteResult(
            suite_name="serde",
            results=[
                EvalResult(
                    task_id="a",
                    passed=True,
                    score=1.0,
                    criteria_scores=[],
                    duration_ms=0,
                )
            ],
            total_duration_ms=10.5,
            total_cost=0.1,
            pass_rate=1.0,
            avg_score=1.0,
            started_at="2024-01-01T00:00:00Z",
            completed_at="2024-01-01T00:00:01Z",
        )
        json_str = json.dumps(sr.to_dict())
        restored = SuiteResult.from_dict(json.loads(json_str))
        assert restored.suite_name == "serde"
        assert restored.results[0].task_id == "a"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_report(self, capsys):
        cli = TinyEvalCLI()
        sr = SuiteResult(
            suite_name="cli-test",
            results=[
                EvalResult("a", True, 1.0, [], 10.0),
                EvalResult("b", False, 0.0, [], 5.0, error="boom"),
            ],
            total_duration_ms=15.0,
            total_cost=0.01,
            pass_rate=0.5,
            avg_score=0.5,
            started_at="2024-01-01T00:00:00Z",
            completed_at="2024-01-01T00:00:01Z",
        )
        cli.report(sr)
        captured = capsys.readouterr()
        assert "cli-test" in captured.out
        assert "50.0%" in captured.out
        assert "✅" in captured.out
        assert "❌" in captured.out

    def test_cli_compare(self, temp_path, capsys):
        cli = TinyEvalCLI()
        baseline = SuiteResult(
            suite_name="x",
            results=[EvalResult("t1", True, 1.0, [], 0)],
            total_duration_ms=0,
            total_cost=0,
            pass_rate=1.0,
            avg_score=1.0,
            started_at="",
            completed_at="",
        )
        current = SuiteResult(
            suite_name="x",
            results=[EvalResult("t1", False, 0.0, [], 0)],
            total_duration_ms=0,
            total_cost=0,
            pass_rate=0.0,
            avg_score=0.0,
            started_at="",
            completed_at="",
        )
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as bf:
            json.dump(baseline.to_dict(), bf)
            bpath = bf.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as cf:
            json.dump(current.to_dict(), cf)
            cpath = cf.name
        try:
            cli.compare(bpath, cpath)
            captured = capsys.readouterr()
            assert "Regression Report" in captured.out
            assert "🔴" in captured.out
        finally:
            os.remove(bpath)
            os.remove(cpath)

    def test_cli_run_suite_verbose(self, capsys):
        cli = TinyEvalCLI()
        cli.run_suite("bug-fix", verbose=True)
        captured = capsys.readouterr()
        assert "PASS" in captured.out or "FAIL" in captured.out


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_p50_p95_latency(self):
        # SuiteResult should track duration_ms per case
        suite = BenchmarkSuite("latency-test")
        for i in range(10):
            case = BenchmarkCase(
                task_id=f"lat_{i}",
                description=f"Latency case {i}",
                task_fn=lambda: {},
                expected_fn=lambda: None,
                rubric=Rubric([
                    Criterion("x", "X", 1.0, lambda t, r: Score(1.0, "ok"))
                ]),
            )
            suite.add_case(case)

        def slow_agent(t):
            time.sleep(0.001)
            return "ok"

        result = suite.run(slow_agent)
        durations = [r.duration_ms for r in result.results]
        assert len(durations) == 10
        # p50
        sorted_d = sorted(durations)
        p50_idx = int(len(sorted_d) * 0.5)
        p50 = sorted_d[p50_idx]
        assert p50 >= 0
        # p95
        p95_idx = int(len(sorted_d) * 0.95)
        p95 = sorted_d[p95_idx]
        assert p95 >= p50

    def test_cost_accumulation(self):
        suite = BenchmarkSuite("cost-test")
        case = BenchmarkCase(
            task_id="costly",
            description="Costly task",
            task_fn=lambda: {},
            expected_fn=lambda: None,
            rubric=Rubric([
                Criterion("x", "X", 1.0, lambda t, r: Score(1.0, "ok"))
            ]),
        )
        suite.add_case(case)

        runner = AgentRunner(
            lambda t: "ok",
            cost_fn=lambda t: 0.03,
        )
        result = runner.run_suite(suite)
        assert result.total_cost == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
