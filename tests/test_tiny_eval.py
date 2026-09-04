"""Tests for tiny-eval."""
import pytest
from tiny_eval import (
    EvalSuite,
    EvalCase,
    EvalResult,
    EvalRun,
    RegressionTracker,
    evaluate,
)


class TestEvalCase:
    def test_basic_run(self):
        case = EvalCase(
            "hello",
            "Should return hello",
            input_fn=lambda: "hello",
            expected_fn=lambda out: 1.0 if out == "hello" else 0.0,
        )
        output, score = case.run()
        assert output == "hello"
        assert score == 1.0

    def test_failing_case(self):
        case = EvalCase(
            "fail",
            "Should fail",
            input_fn=lambda: "wrong",
            expected_fn=lambda out: 1.0 if out == "correct" else 0.0,
        )
        _, score = case.run()
        assert score == 0.0


class TestEvalSuite:
    def test_run_all_pass(self):
        suite = EvalSuite(
            name="test-suite",
            version="1.0.0",
            threshold=0.5,
            cases=[
                EvalCase("a", "case a", lambda: "x", lambda o: 1.0),
                EvalCase("b", "case b", lambda: "y", lambda o: 1.0),
            ],
        )

        def agent(prompt):
            return {"ok": True}

        run = suite.run(agent)
        assert run.total_cases == 2
        assert run.passed_cases == 2
        assert run.failed_cases == 0
        assert run.passed()
        assert run.pass_rate == 1.0

    def test_run_some_fail(self):
        suite = EvalSuite(
            name="partial",
            version="1.0.0",
            threshold=0.7,
            cases=[
                EvalCase("pass", "should pass", lambda: "x", lambda o: 1.0),
                EvalCase("fail", "should fail", lambda: "x", lambda o: 0.0),
            ],
        )

        def agent(prompt):
            return "x"

        run = suite.run(agent)
        assert run.passed_cases == 1
        assert run.failed_cases == 1
        assert not run.passed()
        assert run.total_score == 1.0

    def test_error_handling(self):
        suite = EvalSuite(
            name="error-test",
            version="1.0.0",
            threshold=0.5,
            cases=[
                EvalCase("crash", "should handle errors", lambda: "prompt", lambda o: 1.0),
            ],
        )

        def agent(prompt):
            raise RuntimeError("agent failed")

        run = suite.run(agent)
        assert run.failed_cases == 1
        assert run.results[0].error is not None
        assert "RuntimeError" in run.results[0].error

    def test_suite_to_dict(self):
        suite = EvalSuite(name="x", version="1.0", cases=[], threshold=0.7)
        d = suite.to_dict()
        assert d["name"] == "x"
        assert d["threshold"] == 0.7
        assert d["case_count"] == 0


class TestEvaluate:
    def test_adhoc_evaluate(self):
        def agent(prompt):
            return "hello world"

        result = evaluate(
            agent,
            [
                ("non_empty", "Must return content", lambda o: 1.0 if o else 0.0),
                ("has_hello", "Must say hello", lambda o: 1.0 if "hello" in str(o) else 0.0),
            ],
            suite_name="quick-test",
            threshold=0.5,
        )
        assert result.total_cases == 2
        assert result.passed_cases == 2


class TestRegressionTracker:
    def test_record_and_regression(self, tmp_path):
        path = tmp_path / "runs.json"
        tracker = RegressionTracker(str(path))

        # First run
        run1 = EvalRun(
            id="r1", suite_name="test", suite_version="1.0",
            timestamp="2026-09-01T00:00:00Z", duration_ms=10,
            total_cases=1, passed_cases=1, failed_cases=0,
            total_score=1.0, max_score=1.0, pass_rate=1.0, results=[],
        )
        result1 = tracker.record(run1)
        assert result1.regression is False
        assert result1.delta is None

        # Second run — same score
        run2 = EvalRun(
            id="r2", suite_name="test", suite_version="1.0",
            timestamp="2026-09-02T00:00:00Z", duration_ms=10,
            total_cases=1, passed_cases=1, failed_cases=0,
            total_score=1.0, max_score=1.0, pass_rate=1.0, results=[],
        )
        result2 = tracker.record(run2)
        assert result2.regression is False
        assert result2.delta == 0.0

        # Third run — regression
        run3 = EvalRun(
            id="r3", suite_name="test", suite_version="1.0",
            timestamp="2026-09-03T00:00:00Z", duration_ms=10,
            total_cases=1, passed_cases=0, failed_cases=1,
            total_score=0.0, max_score=1.0, pass_rate=0.0, results=[],
        )
        result3 = tracker.record(run3)
        assert result3.regression is True
        assert result3.delta == -1.0

    def test_best_run(self, tmp_path):
        path = tmp_path / "runs.json"
        tracker = RegressionTracker(str(path))

        for score in [0.5, 0.8, 0.6]:
            run = EvalRun(
                id=f"r{score}", suite_name="bench", suite_version="1.0",
                timestamp="2026-09-01T00:00:00Z", duration_ms=10,
                total_cases=1, passed_cases=1, failed_cases=0,
                total_score=score, max_score=1.0, pass_rate=1.0, results=[],
            )
            tracker.record(run)

        best = tracker.best("bench")
        assert best is not None
        assert best.total_score == 0.8


class TestEvalRun:
    def test_summary(self):
        run = EvalRun(
            id="abc", suite_name="my-suite", suite_version="2.0",
            timestamp="2026-09-01T00:00:00Z", duration_ms=123.4,
            total_cases=5, passed_cases=4, failed_cases=1,
            total_score=3.5, max_score=5.0, pass_rate=0.8, results=[],
        )
        s = run.summary()
        assert "FAIL" in s  # 1 failed case
        assert "3.5/5.0" in s
        assert "123ms" in s

    def test_passed_method(self):
        run = EvalRun(
            id="x", suite_name="y", suite_version="1.0",
            timestamp="t", duration_ms=1,
            total_cases=1, passed_cases=0, failed_cases=1,
            total_score=0.0, max_score=1.0, pass_rate=0.0, results=[],
        )
        assert run.passed() is False

        run2 = EvalRun(
            id="x", suite_name="y", suite_version="1.0",
            timestamp="t", duration_ms=1,
            total_cases=1, passed_cases=1, failed_cases=0,
            total_score=1.0, max_score=1.0, pass_rate=1.0, results=[],
        )
        assert run2.passed() is True
