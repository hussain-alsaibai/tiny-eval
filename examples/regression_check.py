"""
regression_check.py — Example: comparing two eval runs for regression detection.

This demonstrates:
1. Running a suite and recording a baseline
2. Running the suite again after code changes
3. Comparing the two runs to detect regressions
4. Using RegressionReport to surface specific failures
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from tiny_eval import (
    AgentRunner,
    BenchmarkSuite,
    Criterion,
    RegressionReport,
    RegressionTracker,
    Rubric,
    Score,
    SuiteResult,
)


# ---------------------------------------------------------------------------
# Simulate two "runs" at different points in time
# ---------------------------------------------------------------------------

def make_suite_result(suite_name: str, scores: list[tuple[str, float, bool]]) -> SuiteResult:
    """Helper to build a synthetic SuiteResult for demonstration."""
    from tiny_eval import EvalResult

    results = [
        EvalResult(
            task_id=tid,
            passed=passed,
            score=score,
            criteria_scores=[],
            duration_ms=42.0,
        )
        for tid, score, passed in scores
    ]

    avg = sum(r.score for r in results) / len(results) if results else 0.0
    pass_rate = sum(1 for r in results if r.passed) / len(results) if results else 0.0

    now = datetime.now(timezone.utc).isoformat()
    return SuiteResult(
        suite_name=suite_name,
        results=results,
        total_duration_ms=42.0 * len(results),
        total_cost=0.0,
        pass_rate=pass_rate,
        avg_score=avg,
        started_at=now,
        completed_at=now,
    )


# ---------------------------------------------------------------------------
# Demo: compare two runs
# ---------------------------------------------------------------------------

def demo_direct_compare() -> None:
    """Compare two runs using the tracker's compare() method."""
    tracker = RegressionTracker("regression_demo.jsonl")

    # "Baseline" run — all passing, good scores
    baseline_result = make_suite_result("code-quality", [
        ("syntax", 1.0, True),
        ("lint", 1.0, True),
        ("error_handling", 0.9, True),
        ("logging", 0.8, True),
        ("type_hints", 0.7, True),
    ])

    # "Current" run — introduced regressions
    current_result = make_suite_result("code-quality", [
        ("syntax", 1.0, True),
        ("lint", 0.4, False),      # Regressed!
        ("error_handling", 0.5, False),  # Regressed!
        ("logging", 0.0, False),    # New failure!
        ("type_hints", 0.8, True),
    ])

    bid = tracker.record("code-quality", baseline_result, baseline=True)
    cid = tracker.record("code-quality", current_result)

    print("=== Baseline run (id:", bid, ") ===")
    print(f"  Pass rate: {baseline_result.pass_rate:.1%}")
    print(f"  Avg score: {baseline_result.avg_score:.3f}")

    print("\n=== Current run (id:", cid, ") ===")
    print(f"  Pass rate: {current_result.pass_rate:.1%}")
    print(f"  Avg score: {current_result.avg_score:.3f}")

    # Compare
    report = tracker.compare("code-quality", bid, cid)
    print_report(report)


def print_report(report: RegressionReport | None) -> None:
    if report is None:
        print("Could not generate report")
        return

    print("\n" + "=" * 50)
    print("  REGRESSION REPORT")
    print("=" * 50)
    delta = report.delta_score
    icon = "🔴 REGRESSION" if delta < 0 else ("🟢 IMPROVED" if delta > 0 else "⚪ NO CHANGE")
    print(f"  Status: {icon}")
    print(f"  Baseline score : {report.baseline_score:.3f}")
    print(f"  Current score  : {report.current_score:.3f}")
    print(f"  Delta          : {delta:+.3f}")
    print()

    if report.regressions:
        print(f"  📉 Regressed tasks : {', '.join(report.regressions)}")
    if report.new_failures:
        print(f"  ❌ New failures    : {', '.join(report.new_failures)}")
    if report.improved:
        print(f"  ✅ Improved tasks  : {', '.join(report.improved)}")
    if not any([report.regressions, report.new_failures, report.improved]):
        print("  (No individual task changes)")
    print("=" * 50)


def demo_json_compare(baseline_path: str, current_path: str) -> None:
    """Compare two JSON result files directly."""
    with open(baseline_path, encoding="utf-8") as f:
        baseline: SuiteResult = SuiteResult.from_dict(json.load(f))
    with open(current_path, encoding="utf-8") as f:
        current: SuiteResult = SuiteResult.from_dict(json.load(f))

    tracker = RegressionTracker()
    bid = tracker.record(baseline.suite_name, baseline)
    cid = tracker.record(current.suite_name, current)
    report = tracker.compare(baseline.suite_name, bid, cid)
    print_report(report)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        # Compare two JSON files
        demo_json_compare(sys.argv[1], sys.argv[2])
    else:
        # Run the demo
        print("tiny-eval Regression Check — Demo\n")
        demo_direct_compare()
        print("\nTo compare two saved JSON files:")
        print("  python regression_check.py baseline.json current.json")
