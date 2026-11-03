"""
Tests for tiny-eval — AI agent evaluation framework.
Run: python -m pytest tests/ -v
"""

import json
import tempfile
import time

import pytest

import tiny_eval_v100 as tiny_eval
from tiny_eval_v100 import (
    Criterion,
    CriterionResult,
    EvalResult,
    EvalSuite,
    Evaluator,
    Grade,
    bug_fix_suite,
    code_quality_suite,
    eval_run,
    eval_batch,
    security_suite,
)


# ─────────────────────────────────────────────────────────────
# CriterionResult Tests
# ─────────────────────────────────────────────────────────────

class TestCriterionResult:
    def test_to_dict(self):
        cr = CriterionResult(
            name="compiles",
            grade=Grade.PASS,
            score=1.0,
            message="clean build",
            duration_ms=50.0,
        )
        d = cr.to_dict()
        assert d["grade"] == "pass"
        assert d["score"] == 1.0
        assert d["name"] == "compiles"


# ─────────────────────────────────────────────────────────────
# EvalSuite Tests
# ─────────────────────────────────────────────────────────────

class TestEvalSuite:
    def test_all_pass(self):
        suite = EvalSuite("test", [
            Criterion("a", "A", lambda ctx: True),
            Criterion("b", "B", lambda ctx: True),
        ])
        result = suite.evaluate("agent", "run-1", {})
        assert result.overall_pass is True
        assert result.overall_score == 1.0
        assert result.grade == Grade.PASS

    def test_all_fail(self):
        suite = EvalSuite("test", [
            Criterion("a", "A", lambda ctx: False),
        ])
        result = suite.evaluate("agent", "run-1", {})
        assert result.overall_pass is False
        assert result.overall_score == 0.0

    def test_weighted_score(self):
        suite = EvalSuite("test", [
            Criterion("a", "A", lambda ctx: True, weight=1.0),
            Criterion("b", "B", lambda ctx: False, weight=3.0),
        ])
        result = suite.evaluate("agent", "run-1", {})
        # (1*1 + 0*3) / 4 = 0.25
        assert result.overall_score == 0.25

    def test_required_criterion_fails_overrides_score(self):
        suite = EvalSuite("test", [
            Criterion("a", "A", lambda ctx: True, weight=1.0),
            Criterion("b", "B", lambda ctx: True, weight=5.0),
            Criterion("critical", "Critical", lambda ctx: False, required=True),
        ])
        result = suite.evaluate("agent", "run-1", {})
        assert result.overall_pass is False
        assert result.overall_score == 0.0

    def test_criterion_with_message_tuple(self):
        suite = EvalSuite("test", [
            Criterion("a", "A", lambda ctx: (False, "expected True but got False")),
        ])
        result = suite.evaluate("agent", "run-1", {})
        assert result.criteria[0].message == "expected True but got False"

    def test_criterion_exception_marks_fail(self):
        suite = EvalSuite("test", [
            Criterion("a", "A", lambda ctx: (_fail_here(),)),
        ])
        result = suite.evaluate("agent", "run-1", {})
        assert result.criteria[0].grade == Grade.FAIL
        assert "KeyError" in result.criteria[0].message

    def test_pass_threshold(self):
        suite = EvalSuite("test", [
            Criterion("a", "A", lambda ctx: True),
            Criterion("b", "B", lambda ctx: False),
        ], pass_threshold=0.75)
        result = suite.evaluate("agent", "run-1", {})
        assert result.overall_pass is False  # 0.5 < 0.75

    def test_context_passed_correctly(self):
        results = {}
        suite = EvalSuite("test", [
            Criterion("check", "Check ctx",
                      lambda ctx: (results.update(ctx) or True)),
        ])
        suite.evaluate("agent", "run-1", {"key": "value", "num": 42})
        assert results["key"] == "value"
        assert results["num"] == 42


def _fail_here():
    raise KeyError("test exception")


# ─────────────────────────────────────────────────────────────
# Pre-built Suites Tests
# ─────────────────────────────────────────────────────────────

class TestPrebuiltSuites:
    def test_bug_fix_suite(self):
        suite = bug_fix_suite()
        assert suite.name == "bug-fix"
        assert len(suite.criteria) >= 5
        assert suite.pass_threshold == 0.75

        # Happy path
        ctx = {
            "compile_error": None,
            "test_failures": 0,
            "regression_count": 0,
            "secrets_found": 0,
            "crashed": False,
            "duration_s": 30,
        }
        result = suite.evaluate("cursor", "run-1", ctx)
        assert result.overall_pass is True
        assert result.overall_score == 1.0

        # Compile fails
        ctx["compile_error"] = "SyntaxError"
        result = suite.evaluate("cursor", "run-1", ctx)
        assert result.overall_pass is False

    def test_security_suite_strict(self):
        suite = security_suite()
        assert suite.pass_threshold == 1.0  # Must be perfect

        ctx = {"secrets_found": 0, "injection_risk": False, "path_traversal_risk": False}
        result = suite.evaluate("agent", "run-1", ctx)
        assert result.overall_pass is True

        ctx["secrets_found"] = 1
        result = suite.evaluate("agent", "run-1", ctx)
        assert result.overall_pass is False

    def test_code_quality_suite(self):
        suite = code_quality_suite()
        ctx = {
            "lint_errors": 0,
            "type_errors": 0,
            "coverage_percent": 80,
            "static_bugs": 0,
        }
        result = suite.evaluate("agent", "run-1", ctx)
        assert result.overall_pass is True


# ─────────────────────────────────────────────────────────────
# Evaluator Tests
# ─────────────────────────────────────────────────────────────

class TestEvaluator:
    def test_register_and_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ev = Evaluator(tmpdir)
            suite = EvalSuite("test", [
                Criterion("a", "A", lambda ctx: True),
            ])
            ev.register_suite(suite)
            result = ev.run("test", "cursor", "run-1", {})
            assert result.overall_pass is True
            assert result.suite_name == "test"

    def test_run_unknown_suite_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ev = Evaluator(tmpdir)
            with pytest.raises(KeyError, match="No suite registered"):
                ev.run("unknown", "agent", "run-1", {})

    def test_batch_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ev = Evaluator(tmpdir)
            suite = EvalSuite("test", [
                Criterion("a", "A", lambda ctx: True),
            ])
            ev.register_suite(suite)
            contexts = [{}, {}, {}]
            results = ev.run_batch("test", "cursor", contexts)
            assert len(results) == 3
            assert all(r.overall_pass for r in results)

    def test_list_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ev = Evaluator(tmpdir)
            suite = EvalSuite("test", [
                Criterion("a", "A", lambda ctx: True),
            ])
            ev.register_suite(suite)
            ev.run("test", "cursor", "run-1", {})
            ev.run("test", "devin", "run-2", {})

            all_results = ev.list_results()
            assert len(all_results) == 2

            cursor_results = ev.list_results(agent_name="cursor")
            assert len(cursor_results) == 1
            assert cursor_results[0]["agent_name"] == "cursor"

    def test_regression_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ev = Evaluator(tmpdir)
            suite = EvalSuite("test", [
                Criterion("a", "A", lambda ctx: ctx["a"], weight=1.0),
            ])
            ev.register_suite(suite)

            # Historical runs: all pass, so avg is 1.0
            for _ in range(5):
                ev.run("test", "cursor", "run-x", {"a": True})

            # Current run: the criterion fails, so the score drops to 0
            current = ev.run("test", "cursor", "run-new", {"a": False})
            check = ev.regression_check("test", "cursor", current.overall_score)

            assert check["trend"] == "regressing"
            assert check["is_regression"] is True

    def test_regression_no_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ev = Evaluator(tmpdir)
            suite = EvalSuite("test", [Criterion("a", "A", lambda ctx: True)])
            ev.register_suite(suite)
            check = ev.regression_check("test", "cursor", 0.5)
            assert check["trend"] == "unknown"
            assert check["is_regression"] is False


# ─────────────────────────────────────────────────────────────
# EvalResult Tests
# ─────────────────────────────────────────────────────────────

class TestEvalResult:
    def test_to_dict_roundtrip(self):
        result = EvalResult(
            eval_id="abc123",
            suite_name="bug-fix",
            agent_name="cursor",
            run_id="pr-1",
            timestamp=time.time(),
            criteria=[
                CriterionResult("compiles", Grade.PASS, 1.0, "clean build", duration_ms=10),
                CriterionResult("tests_pass", Grade.FAIL, 0.0, "2 failures", duration_ms=50),
            ],
            overall_pass=False,
            overall_score=0.5,
            grade=Grade.FAIL,
            total_duration_ms=60.0,
        )
        d = result.to_dict()
        assert d["overall_pass"] is False
        assert d["criteria"][0]["grade"] == "pass"
        assert d["criteria"][1]["grade"] == "fail"
        assert "timestamp_iso" in d

    def test_summary(self):
        result = EvalResult(
            eval_id="abc",
            suite_name="bug-fix",
            agent_name="cursor",
            run_id="pr-1",
            timestamp=time.time(),
            criteria=[CriterionResult("a", Grade.PASS, 1.0, "ok", duration_ms=10)],
            overall_pass=True,
            overall_score=1.0,
            grade=Grade.PASS,
            total_duration_ms=5.0,
        )
        s = result.summary()
        assert "PASS" in s
        assert "score=100" in s
        assert "bug-fix" in s
