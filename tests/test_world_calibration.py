"""Sanity-check the world: does it reproduce the documented phenomena?"""
import sys, yaml
from collections import Counter, defaultdict
from datetime import datetime, time
sys.path.insert(0, '.')
from data.generator import build, Cause, Rail

rules = yaml.safe_load(open('rules/india_rails.yaml'))
fees = rules['bounce_fees_inr']
fee_rails = {'NACH'}
blocked = [(time(10,0), time(13,0)), (time(17,0), time(21,30))]

w = build(seed=7)
by_dom = defaultdict(lambda: [0,0])
causes = Counter()

for _ in range(w.horizon_days):
    for m in w.due_today():
        # naive: fire once on the due day at 09:30
        when = datetime.combine(w.today, time(9,30))
        o = w.attempt(m.mandate_id, when, blocked_windows=blocked,
                      fee_schedule=fees, fee_rails=fee_rails)
        causes[o.cause.value] += 1
        by_dom[w.today.day][0] += 1
        by_dom[w.today.day][1] += int(o.success)
    w.step()

tot = sum(causes.values()); ok = causes.get('SUCCESS',0)
print(f"attempts={tot}  success_rate={ok/tot:.1%}")
print("\ncauses:")
for c,n in causes.most_common():
    print(f"  {c:26s} {n:5d}  {n/tot:6.1%}")

print(f"\nfees charged to customers: Rs {w.fees_charged_to_customers:,.0f}")

print("\nsuccess rate by day-of-month (>=20 attempts):")
rows=[(d,v[0],v[1]/v[0]) for d,v in sorted(by_dom.items()) if v[0]>=20]
for d,n,r in rows:
    bar = '#'*int(r*40)
    print(f"  day {d:2d}  n={n:4d}  {r:5.1%} {bar}")
early=[r for d,n,r in rows if d<=10]; late=[r for d,n,r in rows if d>=25]
if early and late:
    print(f"\n  days 1-10 mean : {sum(early)/len(early):.1%}")
    print(f"  days 25+  mean : {sum(late)/len(late):.1%}")


# ---------------------------------------------------------------------------
# Assertions
#
# Everything above prints a table a human has to eyeball. Nothing fails if the
# world drifts, which means the calibration claims in FREEZE.md were resting on
# someone remembering to look. Same failure shape as CLAUDE.md 3.9: a check
# that looks like a guarantee and is not.
#
# The targets come from FREEZE.md and are documented industry figures, not
# values tuned to what this world happens to produce.
# ---------------------------------------------------------------------------

import functools as _ft                                              # noqa: E402


@_ft.lru_cache(maxsize=1)
def _calibration():
    """Replay the naive fire-once policy and return the summary statistics.

    Cached: this is a full 400-payer, 6-month rollout and every assertion below
    reads the same one. Without the cache the suite re-simulates the world five
    times over and takes minutes instead of seconds.
    """
    from collections import Counter as _C, defaultdict as _dd
    w2 = build(seed=7)
    by_dom2 = _dd(lambda: [0, 0])
    causes2 = _C()
    for _ in range(w2.horizon_days):
        for m2 in w2.due_today():
            when2 = datetime.combine(w2.today, time(9, 30))
            o2 = w2.attempt(m2.mandate_id, when2, blocked_windows=blocked,
                            fee_schedule=fees, fee_rails=fee_rails)
            causes2[o2.cause.value] += 1
            by_dom2[w2.today.day][0] += 1
            by_dom2[w2.today.day][1] += int(o2.success)
        w2.step()
    tot2 = sum(causes2.values())
    rows2 = [(d, v[0], v[1] / v[0]) for d, v in sorted(by_dom2.items()) if v[0] >= 20]
    early2 = [r for d, n, r in rows2 if d <= 10]
    late2 = [r for d, n, r in rows2 if d >= 25]
    return {
        "attempts": tot2,
        "success_rate": causes2.get("SUCCESS", 0) / tot2,
        "causes": dict(causes2),
        "fees": w2.fees_charged_to_customers,
        "early": sum(early2) / len(early2),
        "late": sum(late2) / len(late2),
    }


def test_blended_success_rate_matches_documented_indian_d2c():
    """FREEZE.md target: 68-74% (Razorpay, D2C blended)."""
    c = _calibration()
    assert 0.68 <= c["success_rate"] <= 0.74, (
        f"blended success rate {c['success_rate']:.1%} has drifted outside the "
        "68-74% band the world is calibrated to")


def test_month_end_degradation_is_present_and_directional():
    """Documented direction only. The magnitude is set by salary_cycle_strength."""
    c = _calibration()
    assert c["early"] > c["late"], (
        f"month-end degradation has vanished: days 1-10 {c['early']:.1%} vs "
        f"days 25+ {c['late']:.1%}")
    assert (c["early"] - c["late"]) > 0.05, "degradation is present but implausibly weak"


def test_insufficient_funds_is_the_dominant_failure():
    """Cause mix: a liquidity problem must dominate, not a technical one."""
    c = _calibration()
    tot = c["attempts"]
    insufficient = c["causes"].get("INSUFFICIENT_FUNDS", 0) / tot
    technical = c["causes"].get("TECHNICAL_DECLINE", 0) / tot
    assert insufficient > 0.10, f"INSUFFICIENT_FUNDS only {insufficient:.1%}"
    assert insufficient > technical * 5, (
        "technical declines rival liquidity failures; this world no longer "
        "represents the problem the project is about")


def test_mandate_revocation_arises_endogenously():
    """>20M revocations/month nationally. It must emerge, not be injected."""
    c = _calibration()
    revoked = c["causes"].get("MANDATE_REVOKED", 0) / c["attempts"]
    assert revoked > 0.02, (
        f"only {revoked:.1%} of attempts hit revoked mandates; the revocation "
        "hazard is no longer producing the documented phenomenon")


def test_the_naive_policy_inflicts_real_money_on_customers():
    """The number the whole project exists to reduce. If it were near zero,
    there would be nothing to optimise and the premise would be wrong."""
    c = _calibration()
    assert c["fees"] > 100_000, (
        f"naive policy inflicted only Rs {c['fees']:,.0f} on customers")


def test_generator_hash_matches_the_freeze_commit():
    """If this fails, results were produced against a different world."""
    from data.generator import World
    assert World.generator_sha256() == (
        "8184e39fbfeb920d7e8fdd97d4b6f2ff260d6baa66d3be3dff370f64a7849d74"), (
        "data/generator.py no longer matches the freeze commit 301dc32. Every "
        "published number is invalid until this is explained in FREEZE.md.")
