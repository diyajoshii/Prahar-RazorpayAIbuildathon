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
