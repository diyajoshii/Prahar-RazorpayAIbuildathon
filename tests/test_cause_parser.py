"""
How well do we classify decline strings, and does the LLM stage earn its place?

Two evaluation sets:

  SEEN     strings drawn from the same pool the world generates. The rules
           table was written with these in view, so it should do well. This
           measures the floor.

  UNSEEN   strings a real bank could plausibly return that the rules table has
           never encountered -- different vendors, different abbreviations,
           Hinglish, truncated logs. Production is full of these. Performance
           here is the only honest measure of whether the LLM stage is worth
           its cost and latency.

Misclassifying a structurally DEAD cause as retryable -- in either the liquidity
or the technical sense -- is the expensive error: it spends capped, fee-bearing
attempts on a mandate that can never be debited until a human re-authorises it.
We report that confusion separately, not just headline accuracy, because a
parser can look respectable on aggregate while making exactly the mistake that
costs the most.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generator import Cause, RAW_MESSAGES
from prahar.causes import CAUSE_CLASS, CauseClass, parse_with_rules
from prahar.llm import CauseParser

# --- held-out strings. None of these appear in RAW_MESSAGES. ----------------
UNSEEN: list[tuple[str, Cause]] = [
    ("A/C BAL NOT ENOUGH FOR DEBIT INSTRUCTION",          Cause.INSUFFICIENT_FUNDS),
    ("khata mein paisa nahi hai - debit fail",            Cause.INSUFFICIENT_FUNDS),
    ("DR AMT > AVL BAL",                                  Cause.INSUFFICIENT_FUNDS),
    ("return reason: funds shortage at drawer bank",      Cause.INSUFFICIENT_FUNDS),
    ("E-MANDATE TERMINATED AT CUSTOMER REQUEST",          Cause.MANDATE_REVOKED),
    ("payer has switched off autopay for this merchant",  Cause.MANDATE_REVOKED),
    ("UMN deregistered on 12-08-2026",                    Cause.MANDATE_REVOKED),
    ("mandate tenure over, please re-register",           Cause.MANDATE_EXPIRED),
    ("subscription authorisation lapsed",                 Cause.MANDATE_EXPIRED),
    ("no umn on file for this vpa",                       Cause.MANDATE_NOT_REGISTERED),
    ("registration incomplete - 2FA never done",          Cause.MANDATE_NOT_REGISTERED),
    ("requested debit 24999 above sanctioned 15000",      Cause.AMOUNT_EXCEEDS_CAP),
    ("value breaches per-txn ceiling on this umn",        Cause.AMOUNT_EXCEEDS_CAP),
    ("customer was not intimated before debit",           Cause.PREDEBIT_NOTICE_FAILED),
    ("advance intimation leg missing",                    Cause.PREDEBIT_NOTICE_FAILED),
    ("remitter psp did not respond in time",              Cause.TECHNICAL_DECLINE),
    ("core banking down for EOD, retry post 2am",         Cause.TECHNICAL_DECLINE),
    ("npci link flap - txn dropped",                      Cause.TECHNICAL_DECLINE),
]

SEEN = [(m, c) for c, msgs in RAW_MESSAGES.items() for m in msgs if c is not Cause.SUCCESS]


def score(name, pairs, fn):
    hits = unknown = fatal = 0
    misses = []
    for raw, truth in pairs:
        p = fn(raw)
        if p.cause is None:
            unknown += 1
            misses.append((raw, truth, "UNKNOWN"))
        elif p.cause is truth:
            hits += 1
        else:
            misses.append((raw, truth, p.cause.value))
            # The expensive error: a structurally dead mandate classified as
            # retryable in ANY sense. Both the liquidity and the technical
            # branches will spend capped, fee-bearing attempts on a mandate
            # that cannot be debited until a human re-authorises it.
            if CAUSE_CLASS[truth] is CauseClass.DEAD and CAUSE_CLASS[p.cause] in (
                CauseClass.RETRYABLE_LIQUIDITY, CauseClass.RETRYABLE_TECHNICAL
            ):
                fatal += 1
    n = len(pairs)
    print(f"\n{name}  n={n}")
    print(f"  correct              {hits:3d}  {hits/n:6.1%}")
    print(f"  unknown (safe)       {unknown:3d}  {unknown/n:6.1%}")
    print(f"  wrong                {n-hits-unknown:3d}  {(n-hits-unknown)/n:6.1%}")
    print(f"  DEAD read as retryable {fatal:3d}  <- spends capped attempts that cannot succeed")
    if misses:
        print("  misses:")
        for raw, truth, got in misses[:8]:
            print(f"    {raw[:48]:50s} truth={truth.value:24s} got={got}")
    return hits / n


if __name__ == "__main__":
    print("=" * 74)
    print("RULES ONLY")
    print("=" * 74)
    score("seen strings", SEEN, parse_with_rules)
    r_unseen = score("held-out strings", UNSEEN, parse_with_rules)

    parser = CauseParser(use_llm=True)
    if not parser.use_llm:
        print("\n" + "=" * 74)
        print("LLM stage SKIPPED - no ANTHROPIC_API_KEY in environment.")
        print("Set it in .env to measure whether the model beats the rules table.")
        print("=" * 74)
    else:
        print("\n" + "=" * 74)
        print(f"RULES + LLM  (model: {parser.model})")
        print("=" * 74)
        score("seen strings", SEEN, parser.parse)
        l_unseen = score("held-out strings", UNSEEN, parser.parse)
        print(f"\nheld-out lift from the LLM stage: {r_unseen:.1%} -> {l_unseen:.1%}")
        print(parser.stats.summary())


# ---------------------------------------------------------------------------
# Assertions
#
# Everything above this line is a reporting script: it prints, and nothing
# fails. That is exactly the failure mode CLAUDE.md 3.9 is about -- a check
# that looks like a guarantee and is not. `pytest` collected nothing from this
# file, so the parser could have regressed to zero silently.
#
# The LLM assertions are opt-in. `pytest tests/` must never make a network call
# on someone else's key, so they skip unless PRAHAR_TEST_LLM=1 is set.
# ---------------------------------------------------------------------------

import os                                                            # noqa: E402

import pytest                                                        # noqa: E402


def _score(pairs, fn):
    hits = unknown = fatal = 0
    for raw, truth in pairs:
        p = fn(raw)
        if p.cause is None:
            unknown += 1
        elif p.cause is truth:
            hits += 1
        elif (CAUSE_CLASS[truth] is CauseClass.DEAD
              and CAUSE_CLASS[p.cause] in (CauseClass.RETRYABLE_LIQUIDITY,
                                           CauseClass.RETRYABLE_TECHNICAL)):
            fatal += 1
    n = len(pairs)
    return hits / n, unknown / n, fatal


def test_rules_handle_every_string_the_world_emits():
    """The floor. The world only emits RAW_MESSAGES, so a miss here means the
    evaluation itself would start seeing spurious UNKNOWNs."""
    acc, _, _ = _score(SEEN, parse_with_rules)
    assert acc == 1.0, f"rules table regressed on seen strings: {acc:.1%}"


def test_rules_alone_cannot_generalise():
    """The entire case for the LLM stage, asserted rather than printed.

    If this ever starts passing at a high rate, someone has quietly widened the
    regex table to cover the held-out set -- which would make the held-out set
    meaningless as a measure of generalisation.
    """
    acc, _, _ = _score(UNSEEN, parse_with_rules)
    assert acc == 0.0, (
        f"rules now score {acc:.1%} on held-out strings. Either the table was "
        "widened to fit them, or the held-out set has leaked into it. Both "
        "destroy the measurement.")


def test_rules_never_call_a_dead_mandate_retryable():
    """The expensive error, on both evaluation sets.

    Misreading a structurally dead cause as retryable spends capped,
    fee-bearing attempts on a mandate that cannot be debited at any hour.
    """
    for name, pairs in (("seen", SEEN), ("held-out", UNSEEN)):
        _, _, fatal = _score(pairs, parse_with_rules)
        assert fatal == 0, f"{name}: {fatal} dead causes classified as retryable"


def test_unseen_strings_are_genuinely_held_out():
    """No held-out string may appear in the pool the world draws from."""
    pool = {m for msgs in RAW_MESSAGES.values() for m in msgs}
    overlap = [s for s, _ in UNSEEN if s in pool]
    assert not overlap, f"held-out strings leaked into RAW_MESSAGES: {overlap}"


@pytest.mark.skipif(os.environ.get("PRAHAR_TEST_LLM") != "1",
                    reason="set PRAHAR_TEST_LLM=1 to spend an API call")
def test_llm_stage_beats_the_rules_table_on_held_out_strings():
    parser = CauseParser(use_llm=True)
    if not parser.use_llm:
        pytest.skip("no API key configured")
    acc, _, fatal = _score(UNSEEN, parser.parse)
    assert acc > 0.60, f"LLM stage scored {acc:.1%} on held-out strings"
    assert fatal == 0, f"{fatal} dead causes classified as retryable by the model"
