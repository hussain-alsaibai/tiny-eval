"""
ci_integration.py — Example: integrating tiny-eval into GitHub Actions CI.

This script is designed to be called from a GitHub Actions workflow step.
It:
1. Loads or defines a benchmark suite
2. Runs the agent against the suite
3. Records results via RegressionTracker
4. Exits with code 1 if a regression is detected (blocking the pipeline)

Usage in GitHub Actions:
  - name: Run agent eval
    run: |
      python examples/ci_integration.py \
        --suite bug-fix \
        --output eval_results.json \
        --tracker-path runs.jsonl \
        --baseline-suite bug-fix

Environment variables (optional):
  GITHUB_RUN_ID — prepended to output for traceability
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tiny_eval import (
    AgentRunner,
    RegressionTracker,
    SuiteResult,
    TinyEvalCLI,
    bug_fix_suite,
    code_quality_suite,
    context_efficiency_suite,
    security_suite,
)


# ---------------------------------------------------------------------------
# Your agent — replace with actual implementation
# ---------------------------------------------------------------------------

def get_agent():
    """
    Return the agent function to evaluate.
    Override this or pass --agent-module to load a custom agent.
    """
    # Default: dummy agent that returns the task unchanged.
    # Replace with your actual agent: LLM call, tool-use loop, etc.
    def default_agent(task):
        return f"Processed: {task}"

    return default_agent


def get_cost_fn():
    """
    Return an optional cost estimation function: (task) -> cost_in_dollars.
    Return 0.0 if not tracking costs.
    """
    # Example: rough estimate per call
    return lambda task: 0.02


# ---------------------------------------------------------------------------
# Suite loader
# ---------------------------------------------------------------------------

def load_suite(name: str):
    """Map a suite name to an actual BenchmarkSuite."""
    loaders = {
        "bug-fix": bug_fix_suite,
        "bug_fix": bug_fix_suite,
        "code-quality": code_quality_suite,
        "code_quality": code_quality_suite,
        "security": security_suite,
        "context-efficiency": context_efficiency_suite,
        "context_efficiency": context_efficiency_suite,
    }
    if name not in loaders:
        raise ValueError(f"Unknown suite: {name!r}. Options: {list(loaders.keys())}")
    return loaders[name]()


# ---------------------------------------------------------------------------
# Main CI entrypoint
# ---------------------------------------------------------------------------

def run_ci(
    suite_name: str,
    output_path: str | None,
    tracker_path: str | None,
    baseline_suite_name: str | None,
    verbose: bool = False,
) -> int:
    """
    Run the evaluation and block CI on regression.

    Returns:
        0  — evaluation passed (no regression)
        1  — regression detected
        2  — evaluation error
    """
    tracker = RegressionTracker(tracker_path) if tracker_path else None
    agent_fn = get_agent()
    cost_fn = get_cost_fn()
    runner = AgentRunner(agent_fn, cost_fn=cost_fn)

    print(f"::group::Loading suite: {suite_name}")
    suite = load_suite(suite_name)
    print(f"  Suite: {suite.name} ({suite.total_cases} cases)")
    print("::endgroup::")

    print("::group::Running evaluation")
    result = runner.run_suite(suite, verbose=verbose)
    print("::endgroup::")

    # Print summary
    cli = TinyEvalCLI()
    cli.report(result)

    # Persist result
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Results written to: {output_path}")

    # Record in tracker
    if tracker:
        run_id = tracker.record(suite_name, result)
        print(f"Recorded as run: {run_id}")

        # Check regression
        if tracker.check_regression(suite_name, result):
            print()
            print("=" * 60)
            print("  🔴  REGRESSION DETECTED — blocking pipeline!")
            print("=" * 60)
            return 1
        else:
            print()
            print("=" * 60)
            print("  ✅  No regression — pipeline may proceed")
            print("=" * 60)
            return 0
    else:
        # No tracker — just return pass/fail based on score threshold
        threshold = 0.7
        if result.avg_score < threshold:
            print(f"  ⚠️  Avg score {result.avg_score:.3f} below threshold {threshold}")
            return 1
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="tiny-eval CI integration script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the bug-fix suite and block on regression
  python ci_integration.py --suite bug-fix --tracker-path runs.jsonl

  # Save results to JSON and check regression
  python ci_integration.py --suite code-quality \\
      --output eval_results.json \\
      --tracker-path runs.jsonl \\
      --baseline-suite code-quality

  # Compare two pre-existing result files
  python ci_integration.py \\
      --compare baseline.json current.json
        """,
    )
    parser.add_argument("--suite", default="bug-fix",
                        help="Suite name to run (default: bug-fix)")
    parser.add_argument("--output", "-o", default=None,
                        help="Path to write JSON results")
    parser.add_argument("--tracker-path", default=None,
                        help="Path to JSONL tracker file")
    parser.add_argument("--baseline-suite", "--baseline", dest="baseline_suite",
                        default=None, help="Suite name to use as baseline source")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CURRENT"),
                        help="Compare two saved JSON result files")
    parser.add_argument("--set-baseline", metavar="RUN_ID",
                        help="Mark a run as the new baseline (requires --tracker-path)")

    args = parser.parse_args()

    # Compare mode
    if args.compare:
        baseline_path, current_path = args.compare
        cli = TinyEvalCLI()
        cli.compare(baseline_path, current_path)
        sys.exit(0)

    # Set baseline mode
    if args.set_baseline:
        if not args.tracker_path:
            print("Error: --tracker-path required with --set-baseline", file=sys.stderr)
            sys.exit(2)
        tracker = RegressionTracker(args.tracker_path)
        ok = tracker.set_baseline(args.suite, args.set_baseline)
        if ok:
            print(f"Run {args.set_baseline} set as baseline for '{args.suite}'")
        else:
            print(f"Run {args.set_baseline} not found for suite '{args.suite}'")
            sys.exit(2)
        sys.exit(0)

    # Run mode
    exit_code = run_ci(
        suite_name=args.suite,
        output_path=args.output,
        tracker_path=args.tracker_path,
        baseline_suite_name=args.baseline_suite,
        verbose=args.verbose,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
