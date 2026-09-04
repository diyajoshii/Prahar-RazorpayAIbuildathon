"""
Shared source scanning for the static invariant tests.

Several invariants are enforced by asserting that certain names never appear in
policy code. Done naively that greps docstrings too, and this codebase's
docstrings *discuss* the very things they must not use -- "we never see
`Payer.balance`" is the module explaining its own constraint, not violating it.

So scanning has to skip docstring bodies, not merely the lines carrying the
delimiter. This helper does that once, so both test modules agree on what
counts as code.
"""

from __future__ import annotations

import pathlib
from typing import Iterator


def code_lines(path: pathlib.Path) -> Iterator[tuple[int, str, str]]:
    """Yield (line_no, raw_line, code_part) for real code lines only.

    Skips comment lines and everything inside a triple-quoted block. The
    tokenizer would be more rigorous, but this stays readable and the failure
    mode of a heuristic here is a false alarm in a test, not a bad number in a
    result.
    """
    in_doc = False
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        ticks = line.count('"""') + line.count("'''")
        if in_doc:
            if ticks:
                in_doc = False
            continue
        if ticks == 1:
            in_doc = True
            continue
        if ticks >= 2 or line.strip().startswith("#"):
            continue
        yield line_no, line, line.split("#", 1)[0]


def scan(package: str, needles: tuple[str, ...]) -> list[str]:
    """Return 'file:line: text' for every code line containing a needle."""
    hits: list[str] = []
    for path in sorted(pathlib.Path(package).glob("*.py")):
        for line_no, line, code in code_lines(path):
            for needle in needles:
                if needle in code:
                    hits.append(f"{path}:{line_no}: {line.strip()}")
    return hits
