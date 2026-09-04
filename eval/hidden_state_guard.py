"""
A tripwire on the world's hidden state.

`CLAUDE.md` §3.2 says the policy may not read `Payer.balance`,
`Payer.salary_day` or `Payer._spend_path`. A rule like that is worth nothing if
nothing enforces it: the violation would be a single plausible-looking line, the
results would improve, and the improvement would be indistinguishable from
progress.

So this module makes the rule executable. It replaces those attributes with
descriptors that inspect the calling frame. The world may read them -- it *is*
the ground truth. Anything under `prahar/` or `baselines/` reading them raises
`HiddenStateViolation` immediately, and the evaluation dies loudly rather than
producing a number nobody can trust.

Tests are deliberately allowed. `tests/test_calendar.py` scores the inferred
salary day against the true one, which is legitimate measurement -- the
distinction being that a test may *know* the answer while a policy may not
*use* it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from data import generator as G

HIDDEN_FIELDS = ("balance", "salary_day", "_spend_path", "monthly_income")

_REPO = Path(__file__).resolve().parent.parent
_FORBIDDEN_DIRS = (_REPO / "prahar", _REPO / "baselines")

_installed = False
_violations: list[str] = []


class HiddenStateViolation(RuntimeError):
    """Policy code read a field the policy is not allowed to see."""


def _caller_is_policy() -> str | None:
    """Return the offending 'file:line' if a policy module is reading, else None.

    Walks a couple of frames rather than one: attribute access can arrive via a
    comprehension or a small helper, and we care about the module that wanted
    the value, not the machinery that fetched it.
    """
    depth = 2
    while depth < 8:
        try:
            frame = sys._getframe(depth)
        except ValueError:
            return None
        path = Path(frame.f_code.co_filename).resolve()
        for bad in _FORBIDDEN_DIRS:
            try:
                path.relative_to(bad)
            except ValueError:
                continue
            return f"{path.name}:{frame.f_lineno} in {frame.f_code.co_name}()"
        # Once we reach the generator or the harness, the read is legitimate.
        if path.name in ("generator.py", "harness.py", "run.py", "sensitivity.py",
                         "trace.py", "hidden_state_guard.py"):
            return None
        depth += 1
    return None


def _make_descriptor(name: str, strict: bool):
    slot = f"__guarded_{name}"

    def getter(self):
        who = _caller_is_policy()
        if who is not None:
            msg = (f"policy code read hidden field Payer.{name} at {who}. "
                   "CLAUDE.md 3.2 forbids this; inferring liquidity from the "
                   "observable signal alone is the problem being solved.")
            _violations.append(msg)
            if strict:
                raise HiddenStateViolation(msg)
        return getattr(self, slot)

    def setter(self, value):
        object.__setattr__(self, slot, value)

    return property(getter, setter)


def install(strict: bool = True) -> None:
    """Arm the tripwire. Idempotent, and safe to call before building a world."""
    global _installed
    if _installed:
        return
    for name in HIDDEN_FIELDS:
        setattr(G.Payer, name, _make_descriptor(name, strict))
    _installed = True


def violations() -> list[str]:
    return list(_violations)


def assert_clean() -> None:
    if _violations:
        raise HiddenStateViolation(
            f"{len(_violations)} hidden-state read(s):\n  " + "\n  ".join(_violations[:5]))


def status() -> str:
    return ("hidden-state guard: ARMED, no violations"
            if _installed and not _violations else
            f"hidden-state guard: {'ARMED' if _installed else 'OFF'}, "
            f"{len(_violations)} violation(s)")


if __name__ == "__main__":
    install(strict=True)
    w = G.build(seed=7)
    p = next(iter(w.payers.values()))
    print("world reading its own state:", f"{p.balance:.2f}", "-> allowed")
    w.step()
    print("world.step() ran -> allowed")
    print(status())
