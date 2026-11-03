"""
tiny-eval CLI entry point.

Usage:
    tiny-eval list [suite]           # List eval results
    tiny-eval run <suite> <agent> <ctx.json>  # Run one eval
    tiny-eval report [opts]          # Generate report
    tiny-eval suites                 # List registered suites
    tiny-eval -h                     # Help
"""

from __future__ import annotations

import json
import sys


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0

    cmd = args[0]

    if cmd == "list":
        suite_filter = args[1] if len(args) > 1 else None
        from tiny_eval_v100 import get_evaluator
        ev = get_evaluator()
        results = ev.list_results(suite_name=suite_filter, limit=30)
        if not results:
            print("No eval results found.")
            return 0
        print(f"{'Eval ID':<10} {'Agent':<18} {'Suite':<18} {'Score':<7} {'Status':<6} {'Time'}")
        print("-" * 80)
        for r in results:
            status = "PASS" if r["overall_pass"] else "FAIL"
            ts = r["timestamp"]
            import time as time_mod
            ts_str = time_mod.strftime("%m-%d %H:%M", time_mod.localtime(ts))
            print(f"{r['eval_id']:<10} {r['agent_name']:<18} {r['suite_name']:<18} "
                  f"{r['overall_score']:.0%}     {status:<6} {ts_str}")

    elif cmd == "run":
        if len(args) < 4:
            print("Usage: tiny-eval run <suite> <agent_name> <run_id> [context.json]")
            print("  context.json is optional. If omitted, reads from stdin.")
            return 1
        suite_name, agent_name, run_id = args[1], args[2], args[3]

        # Read context
        if len(args) > 4:
            with open(args[4]) as f:
                context = json.load(f)
        else:
            context = json.load(sys.stdin)

        from tiny_eval_v100 import eval_run
        result = eval_run(suite_name, agent_name, run_id, context)
        print(result.summary())
        print()
        for c in result.criteria:
            icon = "✅" if c.grade.value == "pass" else "❌"
            print(f"  {icon} [{c.score:.0%}] {c.name}: {c.message}")

    elif cmd == "report":
        from tiny_eval_v100 import get_evaluator
        ev = get_evaluator()
        fmt = "markdown"
        suite_filter = None
        for i, arg in enumerate(args[1:]):
            if arg == "--format" and i + 1 < len(args) - 1:
                fmt = args[i + 2]
            elif not arg.startswith("--"):
                suite_filter = arg
        report = ev.generate_report(suite_name=suite_filter, format=fmt)
        print(report)

    elif cmd == "suites":
        from tiny_eval_v100 import get_evaluator
        ev = get_evaluator()
        print("Registered suites:")
        for name, suite in ev.suites.items():
            print(f"  - {name} ({len(suite.criteria)} criteria, threshold={suite.pass_threshold:.0%})")

    elif cmd == "regression":
        if len(args) < 4:
            print("Usage: tiny-eval regression <suite> <agent_name> <current_score>")
            return 1
        suite_name, agent_name = args[1], args[2]
        current_score = float(args[3])
        from tiny_eval_v100 import get_evaluator
        ev = get_evaluator()
        check = ev.regression_check(suite_name, agent_name, current_score)
        trend_icon = {"improving": "📈", "regressing": "📉", "stable": "➡️"}.get(
            check["trend"], "❓"
        )
        print(f"Regression check for {agent_name} on {suite_name}:")
        print(f"  Trend:    {check['trend']} {trend_icon}")
        print(f"  Delta:    {check['delta']:+.1%}")
        print(f"  Prev avg: {check['previous_avg']:.1%}" if check["previous_avg"] else "  Prev avg: N/A")
        print(f"  Is regression: {check['is_regression']}")
        if check["is_regression"]:
            print("  ⚠️  WARNING: Score regression detected!")
            return 1

    else:
        print(f"Unknown command: {cmd}")
        print("Run 'tiny-eval --help' for usage.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
