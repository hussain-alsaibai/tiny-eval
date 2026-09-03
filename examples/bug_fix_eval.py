"""
bug_fix_eval.py — Example: evaluating an agent on bug-fix tasks.

This demonstrates:
1. Defining custom benchmark cases for a bug-fix workflow
2. Using the AgentRunner with a real-ish agent
3. Recording results with RegressionTracker
"""

from tiny_eval import (
    AgentRunner,
    BenchmarkCase,
    BenchmarkSuite,
    Criterion,
    RegressionTracker,
    Rubric,
    Score,
    bug_fix_suite,
)


# ---------------------------------------------------------------------------
# Simulated agent (replace with your LLM call)
# ---------------------------------------------------------------------------

def mock_agent(task: dict) -> str:
    """
    Simulates an agent that receives a bug description and returns a fix.
    In production, replace this with your LLM API call.
    """
    bug = task.get("bug", "")
    # Simulate a "smart" agent that sometimes introduces secrets or bad syntax
    if "crash" in bug.lower():
        return "def safe_div(a, b):\n    return a / b\n"
    if "secret" in bug.lower():
        return 'def get_api():\n    return "sk-abc123SUPERSECRETKEY"\n'
    return "def add(a, b):\n    return a + b\n"


# ---------------------------------------------------------------------------
# Custom bug-fix evaluation suite
# ---------------------------------------------------------------------------

def arithmetic_bug_suite() -> BenchmarkSuite:
    """Evaluates agents on arithmetic bug-fix tasks."""

    def syntax_check(task: dict, result: str) -> Score:
        try:
            compile(result, "<string>", "exec")
            return Score(1.0, "Valid Python syntax")
        except SyntaxError as e:
            return Score(0.0, f"SyntaxError at line {e.lineno}: {e.msg}")

    def no_regression(task: dict, result: str) -> Score:
        # Ensure the fix doesn't break basic functionality
        try:
            namespace = {}
            exec(result, namespace)
            fn = namespace.get(list(namespace.keys())[-1])
            if fn is None:
                return Score(0.0, "No function defined")
            test_cases = task.get("test_cases", [])
            passed = sum(1 for tc in test_cases if tc["expected"] == fn(*tc["args"]))
            ratio = passed / len(test_cases) if test_cases else 1.0
            return Score(ratio, f"{passed}/{len(test_cases)} test cases passed")
        except Exception as e:
            return Score(0.0, f"Runtime error: {e}")

    rubric = Rubric([
        Criterion("syntax", "Output is syntactically valid Python", 1.0, syntax_check),
        Criterion("no_regression", "Fix passes existing test cases", 1.5, no_regression),
    ])

    suite = BenchmarkSuite(
        "arithmetic-bug-fix",
        description="Evaluates agents on arithmetic bug-fix tasks",
    )

    cases = [
        BenchmarkCase(
            task_id="division_by_zero",
            description="Fix crash on division by zero",
            task_fn=lambda: {
                "bug": "division by zero crash",
                "test_cases": [
                    {"args": (10, 2), "expected": 5},
                    {"args": (10, 0), "expected": None},  # should not crash
                ],
            },
            expected_fn=lambda: None,
            rubric=rubric,
        ),
        BenchmarkCase(
            task_id="add_negative",
            description="Fix incorrect result with negative numbers",
            task_fn=lambda: {
                "bug": "wrong result with negative numbers",
                "test_cases": [
                    {"args": (1, 2), "expected": 3},
                    {"args": (-1, -1), "expected": -2},
                    {"args": (5, -3), "expected": 2},
                ],
            },
            expected_fn=lambda: None,
            rubric=rubric,
        ),
    ]

    for case in cases:
        suite.add_case(case)

    return suite


def main() -> None:
    tracker = RegressionTracker("bug_fix_runs.jsonl")
    runner = AgentRunner(mock_agent)

    # Run custom suite
    print("=== Custom Arithmetic Bug-Fix Suite ===")
    custom_result = runner.run_suite(arithmetic_bug_suite(), verbose=True)
    print(f"\nPass rate: {custom_result.pass_rate:.1%}")
    print(f"Avg score: {custom_result.avg_score:.3f}")

    # Run pre-built suite
    print("\n=== Pre-built Bug-Fix Suite ===")
    prebuilt_result = runner.run_suite(bug_fix_suite(), verbose=True)
    print(f"\nPass rate: {prebuilt_result.pass_rate:.1%}")
    print(f"Avg score: {prebuilt_result.avg_score:.3f}")

    # Record
    run_id = tracker.record("arithmetic-bug-fix", custom_result)
    print(f"\nRecorded run: {run_id}")

    # Check for regression
    if tracker.check_regression("arithmetic-bug-fix", custom_result):
        print("⚠️  REGRESSION DETECTED")
    else:
        print("✅  No regression")


if __name__ == "__main__":
    main()
