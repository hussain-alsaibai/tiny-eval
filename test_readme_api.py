"""The README must document an API that actually exists.

The published README opened with

    from tiny_eval import EvalSuite, EvalCase, evaluate

None of those three names was importable from `tiny_eval`:

* `EvalSuite` lives in `tiny_eval_v100`, not `tiny_eval`
* `EvalCase` and `evaluate` existed in no module at all -- the examples
  were written against an early draft of the API that was replaced by
  `Criterion` / `EvalSuite.evaluate(agent_name, run_id, context)` before
  it was ever merged

`RegressionTracker` was real but the second example mixed it with the
non-existent `suite.run(...)` API from the same draft, so it raised
TypeError even once the import was fixed.

Nothing caught any of this because no test ever read the README.
"""

from __future__ import annotations

import importlib
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(HERE, "README.md")
BLOCK = re.compile(r"```python\n(.*?)```", re.S)
IMPORT_LINE = re.compile(r"^from\s+([\w.]+)\s+import\s+(.+)$", re.M)


def _blocks():
    if not os.path.exists(README):
        pytest.skip("no README.md")
    with open(README, encoding="utf-8") as fh:
        return BLOCK.findall(fh.read())


def _compiles(block):
    """Compile a block, allowing top-level await in examples."""
    try:
        compile(block, "<readme>", "exec")
        return True
    except SyntaxError:
        pass
    indented = "".join(
        "    " + line if line.strip() else line for line in block.splitlines(True)
    )
    try:
        compile("async def _probe():\n" + indented, "<readme>", "exec")
        return True
    except SyntaxError:
        return False


def _import_lines():
    for i, block in enumerate(_blocks()):
        for match in IMPORT_LINE.finditer(block):
            names = []
            for raw in match.group(2).split(","):
                name = raw.strip().split(" as ")[0].strip()
                if name and name.isidentifier():
                    names.append(name)
            if names:
                yield i, match.group(1), names


def test_readme_python_blocks_compile():
    for i, block in enumerate(_blocks()):
        assert _compiles(block), "README block %d does not compile" % i


def test_every_readme_import_resolves():
    sys.path.insert(0, HERE)
    problems = []
    for block, module, names in _import_lines():
        try:
            mod = importlib.import_module(module)
        except Exception as exc:
            problems.append("block %d: cannot import %s (%s)" % (block, module, exc))
            continue
        missing = [n for n in names if not hasattr(mod, n)]
        if missing:
            problems.append("block %d: %s has no %s" % (block, module, ", ".join(missing)))
    assert not problems, "README documents a non-existent API:\n  " + "\n  ".join(problems)


def test_readme_does_not_import_draft_api():
    """Regression: EvalCase/evaluate were in a draft that never shipped.

    The real API is `EvalSuite(name, criteria, pass_threshold)` with
    `Criterion(name, description, check)` and
    `EvalSuite.evaluate(agent_name, run_id, context)`.
    """
    ghosts = {"EvalCase", "evaluate"}
    offenders = [
        (b, module, sorted(ghosts.intersection(names)))
        for b, module, names in _import_lines()
        if ghosts.intersection(names)
    ]
    assert not offenders, (
        "README imports names that exist in no module (EvalCase/evaluate "
        "were a draft; use Criterion + EvalSuite.evaluate instead): %s" % offenders
    )


def test_documented_eval_suite_constructor_is_real():
    """The examples call EvalSuite("name", [Criterion...], pass_threshold=).

    If the constructor changes shape, the README silently becomes wrong
    again, so assert the example's keyword actually exists.
    """
    sys.path.insert(0, HERE)
    from tiny_eval_v100 import EvalSuite

    text = "\n".join(_blocks())
    if "EvalSuite(" not in text:
        pytest.skip("no EvalSuite example")
    suite = EvalSuite("t", [], pass_threshold=0.7)
    assert suite.name == "t"
    assert hasattr(suite, "evaluate")
